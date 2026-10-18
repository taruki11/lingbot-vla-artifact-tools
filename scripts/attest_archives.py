import os
#!/usr/bin/env python3
"""Member-level attestation of downloaded archives against the read-only devzone trees.

WHY THIS EXISTS
/workspace is the only durable volume on this instance and it cannot simultaneously hold the
archive cache (35.69 GiB) and a complete full-parameter DCP (71.33 GiB raw). Retiring an
archive from the cache is admissible ONLY if its content is proven to still exist on the
read-only devzone mount, member by member, and the archive itself is recorded with its pinned
size/SHA256 plus a reproducible re-download command. This script produces that receipt.

It NEVER deletes, moves or modifies anything. Read-only.

Member -> devzone mapping (verified against the archives' central directories):
  dataset/<task>/demo_clean.zip   -> /models/robotwin-persistent/data/demo_clean/<member>
  background_texture.zip          -> /models/robotwin-persistent/assets/<member>
  embodiments.zip                 -> /models/robotwin-persistent/assets/<member>
  objects.zip                     -> /models/robotwin-persistent/assets/<member>

For every archive member this compares the archive's own central-directory (size, CRC32)
against the file on the read-only mount. A missing member or a CRC32/size mismatch fails the
run. Extra files on the mount that no member claims are reported but do not fail (the
devzone tree carries HF downloader residue and __MACOSX AppleDouble junk).

NAME FORMS DIFFER BETWEEN THE TWO SIDES, so members are matched by episode index and directory
alias rather than by literal spelling: the dataset archives store videos unpadded
(`<task>/aloha_agilex/video/episode7.mp4`) while the devzone tree stores them padded
(`aloha_agilex/video/episode_0000007.mp4`), the archives use `instructions/` where the mount
uses `instruction/`, and archive members additionally carry the `<task>/` prefix. Comparing
normalized keys keeps the byte-level CRC32+size requirement intact; it only removes the
naming difference. Absolute, traversing or drive-qualified member names are rejected outright.
"""
import argparse
import json
import pathlib
import re
import zlib
import zipfile

ROOT = pathlib.Path(os.environ.get('ROBOTWIN_PERSISTENT_ROOT', Path.home() / 'robotwin-persistent'))
WORK = pathlib.Path(Path(os.environ.get('LINGBOT_PROJECT_ROOT', Path.cwd())))
CACHE = WORK / 'download-cache' / 'RoboTwin2.0'
MANIFEST = WORK / 'state' / 'dataset-download-manifest.json'

ASSET_TARGET = ROOT / 'assets'
DATASET_TARGET = ROOT / 'data' / 'demo_clean'

REDOWNLOAD = {
    'dataset': 'python3 scripts/fetch_dataset_archives.py --manifest state/dataset-download-manifest.json '
               '--cache download-cache --only-prefix dataset/',
    'assets': 'bash scripts/fetch_asset_archives.sh',
}


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def crc32(path):
    value = 0
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def target_for(row):
    name = row['Path']
    if name.startswith('dataset/'):
        return DATASET_TARGET, 'dataset'
    if name in ('background_texture.zip', 'embodiments.zip', 'objects.zip'):
        return ASSET_TARGET, 'assets'
    return None, None


EPISODE = re.compile(r'episode_?0*(\d+)')
DIRECTORY_ALIASES = {'instructions': 'instruction'}


def normalize(name):
    """Make an archive member name and a devzone path comparable.

    Two naming differences exist between the 50 dataset archives and the extracted devzone
    tree, both purely orthographic — the bytes are identical:
      1. episode files are stored WITHOUT zero padding in the archives
         (`<task>/aloha_agilex/video/episode7.mp4`) and WITH padding on the devzone mount
         (`aloha_agilex/video/episode_0000007.mp4`);
      2. the archives use the plural directory `aloha_agilex/instructions/`, the mount the
         singular `aloha_agilex/instruction/`.
    Dataset archive members additionally carry the `<task>/` prefix, which the mount-relative
    form does not. Normalizing both sides by episode index, directory alias and leading task
    directory makes the two forms comparable; the member is then still verified byte-for-byte
    by size and CRC32. For asset archives this is a no-op.
    """
    parts = [p for p in name.split('/') if p not in ('', '.')]
    if len(parts) > 1 and parts[1] == 'aloha_agilex':
        parts = parts[1:]
    parts = [DIRECTORY_ALIASES.get(p, p) for p in parts]
    return '/'.join(EPISODE.sub(lambda m: f'episode_{int(m.group(1))}', p) for p in parts)


def unsafe_member(name):
    """Absolute, drive-qualified or parent-traversing member names are never admissible.

    `normalize` deliberately does not collapse `..`, so a malicious or corrupt archive cannot
    smuggle a member out of the devzone tree. Such a member fails the archive instead of being
    silently dropped.
    """
    if name.startswith('/') or name.startswith('\\') or ':' in name.split('/')[0]:
        return True
    return '..' in name.split('/')


def attest(row, target, verify_hash):
    archive = CACHE / row['Path']
    entry = dict(path=str(archive), manifest_path=row['Path'], pinned_size=int(row['Size']),
                 pinned_sha256=str(row['Sha256']).lower(), target_root=str(target),
                 redownload=REDOWNLOAD[row['Path'].startswith('dataset/') and 'dataset' or 'assets'])
    if not archive.is_file():
        entry['status'] = 'ABSENT'
        return entry
    size = archive.stat().st_size
    entry['size'] = size
    if size != int(row['Size']):
        entry['status'] = 'SIZE_MISMATCH'
        return entry
    if verify_hash:
        sha = digest(archive)
        entry['sha256'] = sha
        if sha != str(row['Sha256']).lower():
            entry['status'] = 'SHA256_MISMATCH'
            return entry
        entry['sha256_verified'] = True
    with zipfile.ZipFile(archive) as zf:
        members = [(info.filename, info.file_size, info.CRC) for info in zf.infolist() if not info.is_dir()]
    entry['members'] = len(members)
    entry['member_bytes'] = sum(m[1] for m in members)
    # Dataset archives root their members at `<task>/`, so the on-disk lookup happens under
    # `<dataset root>/<task>/` while the comparison key has that same `<task>/` stripped.
    task = row['Path'].split('/')[1] + '/' if row['Path'].startswith('dataset/') else ''
    lookup = target / task.rstrip('/') if task else target
    # Index the devzone subtree by NORMALIZED name -> real path. The normalized key is NOT a
    # usable path (it collapses `episode_0000007.mp4` to `episode_7.mp4`), so every lookup must
    # go through this index; using the key directly as a path silently reports valid members
    # as missing.
    #
    # The scan is limited to this archive's OWN top-level directories, because the asset
    # archives share one root (`assets/`): indexing the whole root would attribute another
    # archive's files to this one and inflate `extra_on_disk` into nonsense.
    #
    # A dataset archive roots its members at `<task>/`, and `lookup` already IS
    # `<dataset root>/<task>`, so the task component must be stripped before the first
    # remaining component can name a directory under `lookup`.
    tops = set()
    for name, _, _ in members:
        parts = [p for p in name.split('/') if p not in ('', '.')]
        if task and parts and parts[0] == task.rstrip('/'):
            parts = parts[1:]
        if parts:
            tops.add(parts[0])
    on_disk = {}
    for top in sorted(tops):
        scope = lookup / top
        if not scope.is_dir():
            continue
        for p in scope.rglob('*'):
            if p.is_file():
                on_disk[normalize(task + str(p.relative_to(lookup)))] = p
    missing, mismatch, unsafe = [], [], []
    claimed = set()
    for name, member_size, member_crc in members:
        if unsafe_member(name):
            unsafe.append(name)
            continue
        key = normalize(name)
        claimed.add(key)
        candidate = on_disk.get(key)
        if candidate is None:
            missing.append(name)
            continue
        if candidate.stat().st_size != member_size or crc32(candidate) != member_crc:
            mismatch.append(f'{name} -> {candidate}')
    entry['attested'] = len(members) - len(missing) - len(mismatch) - len(unsafe)
    entry['missing'] = len(missing)
    entry['mismatch'] = len(mismatch)
    entry['unsafe_members'] = len(unsafe)
    entry['missing_sample'] = missing[:5]
    entry['mismatch_sample'] = mismatch[:5]
    entry['extra_on_disk'] = len(set(on_disk) - claimed)
    entry['extra_scope'] = sorted(tops)
    entry['name_normalization'] = ('episode index; instructions->instruction alias; '
                                   'leading <task>/ stripped for dataset archives')
    entry['status'] = 'OK' if not missing and not mismatch and not unsafe else 'FAILED'
    if entry['status'] == 'OK':
        del entry['missing_sample']
        del entry['mismatch_sample']
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['dataset', 'assets', 'all'], default='all')
    ap.add_argument('--skip-hash', action='store_true', help='skip the archive-level sha256 (members are still CRC-checked)')
    ap.add_argument('--out', default=str(WORK / 'state' / 'archive-attestation.json'))
    args = ap.parse_args()

    rows = json.loads(MANIFEST.read_text())
    selected = []
    for row in rows:
        target, kind = target_for(row)
        if target is None or (args.only != 'all' and kind != args.only):
            continue
        selected.append((row, target))

    results = []
    failures = 0
    for row, target in selected:
        entry = attest(row, target, verify_hash=not args.skip_hash)
        results.append(entry)
        if entry['status'] != 'OK':
            failures += 1
        print(f"{entry['status']:16s} {entry['manifest_path']} members={entry.get('members', '-')} "
              f"attested={entry.get('attested', '-')} missing={entry.get('missing', '-')} "
              f"mismatch={entry.get('mismatch', '-')}", flush=True)

    payload = dict(kind=args.only, generated_at_unix=None, archives=results,
                   archives_ok=sum(1 for r in results if r['status'] == 'OK'),
                   archives_failed=failures,
                   scope='every archive member compared by size and CRC32 against the read-only devzone tree',
                   attestation='central-directory CRC32 and file_size vs the extracted read-only mount')
    import time
    payload['generated_at_unix'] = int(time.time())
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f'ATTESTATION archives={len(results)} ok={payload["archives_ok"]} failed={failures} out={args.out}')
    print('ATTEST_OK' if not failures else 'ATTEST_FAILED')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
