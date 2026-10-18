"""Archive a completed DCP once; decode and hash every member without a restore copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile

from archive_checkpoint import sha


def verify_archive(path, expected):
    decoder = subprocess.Popen(['zstd', '-d', '-q', '-c', str(path)], stdout=subprocess.PIPE)
    actual = {}
    try:
        with tarfile.open(fileobj=decoder.stdout, mode='r|') as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                assert not name.is_absolute() and '..' not in name.parts, 'Unsafe member path'
                if member.isdir():
                    continue
                assert member.isfile(), 'Links and special files are not checkpoint members'
                key = str(name)
                assert key in expected and key not in actual, 'Unexpected/duplicate member'
                digest = hashlib.sha256()
                stream = archive.extractfile(member)
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
                actual[key] = digest.hexdigest()
        # Read through the compressed footer so zstd also checks the frame checksum.
        for block in iter(lambda: decoder.stdout.read(8 << 20), b''):
            assert not any(block), 'Unexpected data after tar end'
        assert decoder.wait(timeout=30) == 0, 'Archive decompression failed'
        assert actual == expected, 'Decoded members differ from the training receipt'
        return actual
    finally:
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait(timeout=30)


def archive_checkpoint(checkpoint, expected, destination):
    files = {str(p.relative_to(checkpoint)):p for p in checkpoint.rglob('*') if p.is_file()}
    assert set(files) == set(expected)
    assert not any(p.is_symlink() for p in checkpoint.rglob('*'))
    assert not destination.resolve().is_relative_to(checkpoint.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    assert not destination.exists() and not partial.exists(), 'Never overwrite an archive'
    raw_bytes = sum(p.stat().st_size for p in files.values())
    assert shutil.disk_usage(destination.parent).free > raw_bytes * 1.01 + 5 * 1024**3
    with partial.open('xb') as stream:
        tar = subprocess.Popen(['tar', '-C', str(checkpoint), '-cf', '-', '.'], stdout=subprocess.PIPE)
        try:
            compressed = subprocess.run(['zstd', '-1', '-T8', '-q', '-c'], stdin=tar.stdout,
                                        stdout=stream, timeout=7200)
        finally:
            tar.stdout.close()
            if tar.poll() is None and 'compressed' not in locals():
                tar.terminate()
        assert tar.wait(timeout=30) == compressed.returncode == 0, 'Compression failed'
        stream.flush()
        os.fsync(stream.fileno())
    actual = verify_archive(partial, expected)
    digest = sha(partial)
    # This rename only finalizes the new temporary file created above.
    assert not destination.exists()
    partial.rename(destination)
    return dict(status='completed', source_checkpoint=str(checkpoint), archive=str(destination),
                archive_bytes=destination.stat().st_size, archive_sha256=digest,
                verified_files=len(actual), file_sha256=actual,
                restore_scope='All tar members decoded and SHA256-matched to the training receipt; no second full extraction directory, no new optimizer/GPU load claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    assert args.run and all(c.isalnum() or c in '_-' for c in args.run)
    work = Path(Path(os.environ.get('LINGBOT_PROJECT_ROOT', Path.cwd())))
    folder = work / 'runs' / args.run
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['status'] == 'completed' and receipt['stage'] in ('smoke', 'train')
    checkpoint = Path(receipt['checkpoint'])
    assert checkpoint.resolve().is_relative_to(Path(os.environ.get('LINGBOT_RUNTIME_RUNS', Path.home() / 'lingbot-runtime' / 'runs')) / args.run / 'training/checkpoints')
    proof = folder / 'stream-backup-verification.json'
    assert not proof.exists(), 'Never replace a historical backup proof'
    result = archive_checkpoint(checkpoint, receipt['checkpoint_hashes'], args.destination)
    result['persistent_destination'] = args.destination.resolve().is_relative_to(work)
    with proof.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
