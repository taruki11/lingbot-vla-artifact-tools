"""Persist a certified HF export after checking the archive and every restored file."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from archive_checkpoint import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-run', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    assert args.export_run and all(c.isalnum() or c in '_-' for c in args.export_run)
    work = Path(Path(os.environ.get('LINGBOT_PROJECT_ROOT', Path.cwd())))
    run = work / 'runs' / args.export_run
    receipt_path = run / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['stage'] == 'export' and receipt['status'] == 'completed'
    source = Path(receipt['model'])
    assert source == Path(os.environ.get('LINGBOT_RUNTIME_RUNS', Path.home() / 'lingbot-runtime' / 'runs')) / args.export_run / 'hf'
    assert not any(p.is_symlink() for p in source.rglob('*'))
    expected = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    assert all(expected[name] == digest for name, digest in receipt['model_hashes'].items())
    assert expected['norm_stats.json'] == receipt['norm_sha256']
    assert expected['training-config.yaml'] == receipt['training_config_sha256']
    raw_size = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    assert shutil.disk_usage(source).free > 2 * raw_size + 5 * 1024**3
    scratch = source.parent / 'hf-backup-validation'
    scratch.mkdir(exist_ok=False)
    staging = scratch / 'hf.tar.zst'
    with staging.open('xb') as stream:
        tar = subprocess.Popen(['tar', '-C', str(source), '-cf', '-', '.'], stdout=subprocess.PIPE)
        compressed = subprocess.run(['zstd', '-3', '-T8', '-q', '-c'], stdin=tar.stdout, stdout=stream)
        tar.stdout.close()
        assert tar.wait() == compressed.returncode == 0
        stream.flush(); os.fsync(stream.fileno())
    digest = sha(staging)
    destination = args.destination.resolve()
    assert destination.is_relative_to(work / 'candidates')
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.partial')
    assert not destination.exists() and not partial.exists(), 'Never overwrite an archive'
    assert shutil.disk_usage(destination.parent).free > staging.stat().st_size + 2 * 1024**3, 'Need actual compressed size plus 2 GiB margin'
    with partial.open('xb') as target, staging.open('rb') as original:
        shutil.copyfileobj(original, target, 8 << 20)
        target.flush(); os.fsync(target.fileno())
    assert partial.stat().st_size == staging.stat().st_size and sha(partial) == digest
    subprocess.run(['zstd', '-t', '-q', str(partial)], check=True)
    restored = scratch / 'restored'
    restored.mkdir()
    decompressor = subprocess.Popen(['zstd', '-d', '-q', '-c', str(partial)], stdout=subprocess.PIPE)
    extracted = subprocess.run(['tar', '-xf', '-', '-C', str(restored)], stdin=decompressor.stdout)
    decompressor.stdout.close()
    assert decompressor.wait() == extracted.returncode == 0
    actual = {str(p.relative_to(restored)): sha(p) for p in sorted(restored.rglob('*')) if p.is_file()}
    assert actual == expected, 'Restored HF contents differ from the certified source'
    partial.rename(destination)
    proof = dict(status='completed', export_run=args.export_run, export_receipt_sha256=sha(receipt_path),
                 source=str(source), archive=str(destination), archive_bytes=destination.stat().st_size,
                 archive_sha256=digest, verified_files=len(actual), file_sha256=actual,
                 norm_sha256=receipt['norm_sha256'], persistent_destination=True,
                 scope='Complete persisted HF archive and restored file hashes verified; GPU load is evidenced by the corresponding evaluation.')
    (run / 'persistence-verification.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
