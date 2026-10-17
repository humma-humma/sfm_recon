"""Check the portable Pages artifact, including assets and project-relative links."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('href', 'src', 'poster') and value:
                self.urls.append(value)


def verify(root):
    root = root.resolve()
    for path in root.rglob('*'):
        assert not path.is_symlink(), f'Artifact contains a symbolic link: {path}'
        if path.suffix != '.html':
            continue
        parser = Links()
        parser.feed(path.read_text(encoding='utf-8'))
        for link in parser.urls:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            assert not url.path.startswith('/'), f'Breaks project-site base path: {link}'
            target = (path.parent / unquote(url.path)).resolve()
            assert target.is_relative_to(root), f'Escapes artifact: {link}'
            if target.is_dir():
                target /= 'index.html'
            assert target.is_file(), f'Missing asset: {link}'
    manifest = json.loads((root / 'media-manifest.json').read_text())
    for item in manifest['videos']:
        for name in [item['path'], *item.get('aliases', [])]:
            p = root / name
            digest = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            assert p.stat().st_size == item['bytes'] and digest.hexdigest() == item['sha256'], name
    data = root / 'assets/gnt_finalization/scene15_splat'
    assert json.loads((data / 'comparison_progress.json').read_text())['status'] == 'complete'
    assert json.loads((data / 'results.json').read_text())['anchor_count'] == 2198
    home = (root / 'index.html').read_text(encoding='utf-8')
    assert 'Full video portfolio' not in home and '<footer>' not in home
    assert home.index('id="project-heading"') < home.index('id="datasets"') < home.index('id="runtime-table"')
    measurements = json.loads((root / 'public_data/homepage-results.json').read_text())
    scene_table = home.split('id="scene-quality-table"')[1].split('</table>')[0]
    for scores in measurements['scene']['outdoor_gap_scores'].values():
        assert scores['count'] == 89
        assert f'{scores["gap_psnr"]:.2f}' in scene_table and f'{scores["gap_ssim"]:.4f}' in scene_table
    object_table = home.split('id="object-quality-table"')[1].split('</table>')[0]
    for item in measurements['objects'].values():
        scores = item['heldout_photos']
        assert f'{scores["psnr"]:.2f}' in object_table and f'{scores["ssim"]:.4f}' in object_table
    assert not any(root.rglob('PROJECT_FULL_WRITEUP*')), 'Local report must not be published'
    print('Passed: portable links, video hashes, homepage metrics and layout, no junctions or local report')


if __name__ == '__main__':
    verify(Path(sys.argv[1] if len(sys.argv) > 1 else 'dist/demo'))
