"""Package the final route demo without following local artifact junctions.

By default read the existing local videos. In Actions, --assets-dir points at
the flat release downloads. Every video is checked against the versioned manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web_demo'


def build(output, assets_dir=None):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to avoid publishing stale files')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('index.html', 'demo.js', 'demo.css', 'portfolio.html', 'gnt-scene-15fps-splats.html'):
        text = (WEB / name).read_text(encoding='utf-8')
        if name == 'portfolio.html':
            text = text.replace('href="gnt-balanced-all.html"',
                                'href="https://github.com/humma-humma/sfm_recon#verified-results"')
        (output / name).write_text(text, encoding='utf-8')
    (output / '.nojekyll').touch()
    manifest = json.loads((WEB / 'media-manifest.json').read_text())
    total = 0
    for item in manifest['videos']:
        source = assets_dir / item['release_asset'] if assets_dir else WEB / item['path']
        digest = hashlib.sha256()
        with source.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if source.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Video integrity check failed: {source}')
        target = output / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        for alias in item.get('aliases', []):
            alias_target = output / alias
            alias_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, alias_target)
        total += item['bytes']
    posters = output / 'assets/portfolio'
    posters.mkdir(parents=True)
    for name in ('box', 'milk', 'shoe', 'scene_motion', 'scene_splats'):
        shutil.copyfile(WEB / 'assets/portfolio' / (name + '.jpg'), posters / (name + '.jpg'))
    reports = output / 'assets/gnt_finalization/scene15_splat'
    for name in ('results.json', 'comparison_progress.json'):
        shutil.copyfile(WEB / 'public_data' / name, reports / name)
    shutil.copyfile(WEB / 'media-manifest.json', output / 'media-manifest.json')
    print(f'Packaged {len(manifest["videos"])} verified videos ({total / 1e6:.1f} MB) into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/demo')
    parser.add_argument('--assets-dir', type=Path)
    args = parser.parse_args()
    build(args.output, args.assets_dir)
