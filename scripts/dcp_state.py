#!/usr/bin/env python3
"""Read a torch.distributed.checkpoint (DCP) directory and report its state.

Read-only. Purpose: prove WITHOUT training that a checkpoint really carries optimizer,
scheduler, dataloader and RNG state -- the four things the project requires before any
continuation claim (user m01976). Two independent sources are used:

1. extra_state/extra_state_rank_<n>.pt -- the trainer's own dict, written at
   tasks/vla/train_lingbotvla.py:1117-1145 with keys global_step, lr_scheduler,
   train_dataloader, environ_meter, torch_rng_state.
2. <sub>/.metadata via torch.distributed.checkpoint.FileSystemReader.read_metadata() --
   the DCP planner's manifest of every stored tensor, which is where AdamW moments
   (exp_avg / exp_avg_sq / step) can be counted without loading 51 GB of shards.

The .metadata files are NOT torch pickles (torch.load fails with "Invalid magic number");
they must be read through FileSystemReader. That was established by measurement.

Usage: dcp_state.py <dcp-dir> [--json <out>]
Prints a JSON object and exits non-zero if the checkpoint is incomplete.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def extra_state(dcp: pathlib.Path):
    """Load every extra_state_rank_*.pt and reduce it to comparable scalars."""
    import torch

    rows = []
    for path in sorted((dcp / 'extra_state').glob('extra_state_rank_*.pt')):
        value = torch.load(path, map_location='cpu', weights_only=False)
        scheduler = value.get('lr_scheduler') or {}
        dataloader = value.get('train_dataloader') or {}
        rng = value.get('torch_rng_state')
        rows.append(dict(
            rank=path.stem.replace('extra_state_rank_', ''),
            keys=sorted(value),
            global_step=value.get('global_step'),
            lr_last_epoch=scheduler.get('last_epoch'),
            lr_step_count=scheduler.get('_step_count'),
            lr_last=([float(v) for v in scheduler.get('_last_lr', [])] if scheduler.get('_last_lr') is not None else None),
            lr_lambdas=len(scheduler.get('lr_lambdas') or []),
            dataloader_keys=sorted(dataloader),
            dataloader_steps_since_snapshot=dataloader.get('_steps_since_snapshot'),
            rng_shape=(list(rng.shape) if hasattr(rng, 'shape') else None),
            has_environ_meter='environ_meter' in value,
        ))
    return rows


def metadata(dcp: pathlib.Path):
    """Read the DCP planner manifests for every sub-directory."""
    from torch.distributed.checkpoint import FileSystemReader

    result = {}
    for sub in ('model', 'optimizer'):
        path = dcp / sub
        if not path.is_dir():
            continue
        entries = FileSystemReader(str(path)).read_metadata().state_dict_metadata
        result[sub] = dict(
            entries=len(entries),
            exp_avg=len([k for k in entries if k.endswith('.exp_avg')]),
            exp_avg_sq=len([k for k in entries if k.endswith('.exp_avg_sq')]),
            step=len([k for k in entries if k.endswith('.step')]),
            sample=sorted(entries)[:3],
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dcp', type=pathlib.Path)
    parser.add_argument('--json', type=pathlib.Path)
    args = parser.parse_args()
    dcp = args.dcp.resolve()
    if not dcp.is_dir():
        print(f'DCP_STATE_FAILED not a directory: {dcp}')
        return 2

    shards = {sub: sorted((dcp / sub).glob('*.distcp')) for sub in ('model', 'optimizer') if (dcp / sub).is_dir()}
    value = dict(
        dcp=str(dcp),
        shards={sub: len(paths) for sub, paths in shards.items()},
        bytes={sub: sum(p.stat().st_size for p in paths) for sub, paths in shards.items()},
        has_metadata={sub: (dcp / sub / '.metadata').is_file() for sub in ('model', 'optimizer')},
        extra_state=extra_state(dcp),
        metadata=metadata(dcp),
    )
    problems = []
    if not value['shards'].get('model') or not value['shards'].get('optimizer'):
        problems.append('missing model or optimizer shards')
    if not all(value['has_metadata'].values()):
        problems.append('missing DCP .metadata')
    for sub in ('model', 'optimizer'):
        entry = value['metadata'].get(sub) or {}
        if not entry.get('entries'):
            problems.append(f'{sub}: no metadata entries')
    optimizer = value['metadata'].get('optimizer') or {}
    if not optimizer.get('exp_avg') or not optimizer.get('exp_avg_sq'):
        problems.append('optimizer metadata carries no AdamW moments (exp_avg/exp_avg_sq)')
    if not value['extra_state']:
        problems.append('no extra_state_rank_*.pt files')
    for row in value['extra_state']:
        if row['global_step'] is None:
            problems.append(f"rank {row['rank']}: no global_step")
        if not row['lr_lambdas']:
            problems.append(f"rank {row['rank']}: scheduler state has no lr_lambdas")
        if row['rng_shape'] is None:
            problems.append(f"rank {row['rank']}: no torch_rng_state")
        if not row['has_environ_meter']:
            problems.append(f"rank {row['rank']}: no environ_meter")
    steps = sorted({row['global_step'] for row in value['extra_state']})
    if len(steps) > 1:
        problems.append(f'ranks disagree on global_step: {steps}')
    value['global_step'] = steps[0] if len(steps) == 1 else None
    value['problems'] = problems

    if args.json:
        args.json.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k not in ('extra_state', 'metadata')}, indent=2))
    for row in value['extra_state']:
        print(f"  rank {row['rank']}: global_step={row['global_step']} last_epoch={row['lr_last_epoch']} "
              f"step_count={row['lr_step_count']} lr_last={row['lr_last']} rng={row['rng_shape']}")
    print(f"  metadata: {json.dumps(value['metadata'])}")
    if problems:
        print('DCP_STATE_FAILED ' + '; '.join(problems))
        return 2
    print(f"DCP_STATE_OK global_step={value['global_step']} shards={value['shards']} bytes={value['bytes']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
