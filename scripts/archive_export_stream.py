"""Archive a completed HF export once using the verified streaming archive utility."""
import os
import argparse
import json
from pathlib import Path

from archive_checkpoint import sha
from archive_checkpoint_stream import archive_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--export-run', required=True)
parser.add_argument('--destination', required=True, type=Path)
args = parser.parse_args()
assert args.export_run and all(c.isalnum() or c in '_-' for c in args.export_run)
folder = Path(os.environ.get('LINGBOT_RUNS_DIR', Path(os.environ.get('LINGBOT_PROJECT_ROOT', Path.cwd())) / 'runs')) / args.export_run
receipt_path = folder / 'receipt.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['stage'] == 'export' and receipt['status'] == 'completed'
source = Path(receipt['model'])
assert source == Path(os.environ.get('LINGBOT_RUNTIME_RUNS', Path.home() / 'lingbot-runtime' / 'runs')) / args.export_run / 'hf'
proof_path = folder / 'stream-backup-verification.json'
assert not proof_path.exists()
expected = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
assert all(expected[name] == digest for name, digest in receipt['model_hashes'].items())
assert expected['norm_stats.json'] == receipt['norm_sha256']
assert expected['training-config.yaml'] == receipt['training_config_sha256']
proof = archive_checkpoint(source, expected, args.destination)
proof.update(export_run=args.export_run, export_receipt_sha256=sha(receipt_path),
             norm_sha256=receipt['norm_sha256'], persistent_destination=False)
with proof_path.open('x') as stream:
    json.dump(proof, stream, indent=2)
print(json.dumps(proof, indent=2), flush=True)
