"""Archive one certified DCP, restore it, and compare every file with its receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    work = Path(Path(os.environ.get('LINGBOT_PROJECT_ROOT', Path.cwd())))
    assert args.run.isalnum() or all(c.isalnum() or c in '_-' for c in args.run)
    run = work / 'runs' / args.run
    receipt = json.loads((run / 'receipt.json').read_text())
    assert receipt['status'] == 'completed' and receipt['stage'] in ('smoke', 'train')
    checkpoint = Path(receipt['checkpoint'])
    scratch = checkpoint.parent.parent / 'backup-validation'
    scratch.mkdir(exist_ok=False)
    staging = scratch / 'checkpoint.tar.zst'
    with staging.open('xb') as out:
        tar = subprocess.Popen(['tar', '-C', str(checkpoint), '-cf', '-', '.'], stdout=subprocess.PIPE)
        compressed = subprocess.run(['zstd', '-3', '-T8', '-q', '-c'], stdin=tar.stdout, stdout=out)
        tar.stdout.close()
        assert tar.wait() == compressed.returncode == 0, 'Compression failed'
        out.flush()
        os.fsync(out.fileno())
    archive_sha = sha(staging)
    destination = args.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists() and not destination.with_suffix('.partial').exists()
    assert shutil.disk_usage(destination.parent).free > staging.stat().st_size + 5 * 1024**3
    partial = destination.with_suffix('.partial')
    shutil.copyfile(staging, partial)
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    assert sha(partial) == archive_sha
    partial.rename(destination)
    print('ARCHIVE_COPIED', destination, destination.stat().st_size, archive_sha, flush=True)
    subprocess.run(['zstd', '-t', '-q', str(destination)], check=True)
    restored = scratch / 'restored'
    restored.mkdir()
    decompressor = subprocess.Popen(['zstd', '-d', '-q', '-c', str(destination)], stdout=subprocess.PIPE)
    extracted = subprocess.run(['tar', '-xf', '-', '-C', str(restored)], stdin=decompressor.stdout)
    decompressor.stdout.close()
    assert decompressor.wait() == extracted.returncode == 0
    actual = {str(p.relative_to(restored)): sha(p) for p in sorted(restored.rglob('*')) if p.is_file()}
    assert actual == receipt['checkpoint_hashes'], 'Restored DCP differs from certified checkpoint'
    result = dict(status='completed', source_checkpoint=str(checkpoint), archive=str(destination),
                  archive_bytes=destination.stat().st_size, archive_sha256=archive_sha,
                  restored=str(restored), verified_files=len(actual),
                  persistent_destination=destination.resolve().is_relative_to(work),
                  restore_scope='all restored files SHA256 match the certified checkpoint; GPU reload is verified separately')
    (run / 'backup-verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
