"""Verify that released experiment files match their recorded Kaya hashes."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance/kaya_sources.json').read_text())
    checked = 0
    for source in manifest['sources'].values():
        directory = root / source['repository_path']
        for relative, expected in source['files'].items():
            path = directory / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise SystemExit(f'Source hash mismatch: {path.relative_to(root)}')
            checked += 1
    frozen = json.loads((root / 'code_manifest.json').read_text())
    for relative, expected in frozen.items():
        if hashlib.sha256((root / 'code' / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Experiment hash mismatch: {relative}')
    print(f'Verified {checked} source files and {len(frozen)} experiment hashes.')


if __name__ == '__main__':
    main()
