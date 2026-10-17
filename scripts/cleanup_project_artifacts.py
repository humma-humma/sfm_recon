"""Manifest-driven cleanup of explicitly scoped, disposable project artifacts.

Run without arguments to build the plan; --execute applies that exact plan.
Never traverses junctions, deletes tracked files, or touches original datasets.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/project_cleanup'
ART = Path.home() / 'sfm_reconstruction_artifacts'
TEMP = Path.home() / 'AppData/Local/Temp/sfm_lightweight_methods_v1'
ALLOWED = [ROOT, ART, TEMP]

def key(p):
    return os.path.normcase(os.path.abspath(p))

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def safe(p):
    absolute = p.absolute()
    base = next((r for r in ALLOWED if absolute.is_relative_to(r)), None)
    if base is None:
        raise ValueError(f'Outside cleanup roots: {p}')
    for part in [absolute, *absolute.parents]:
        if part == base.parent:
            break
        if part.lstat().st_file_attributes & 1024:
            raise ValueError(f'Reparse point: {part}')
    if absolute.resolve() != absolute:
        raise ValueError(f'Unexpected resolution: {p}')

def evidence():
    m = json.loads((ROOT / 'docs/PROJECT_EVIDENCE_MANIFEST.json').read_text())
    records = []
    for v in m.values():
        for rec in v if isinstance(v, list) else [v]:
            if isinstance(rec, dict) and 'path' in rec and 'sha256' in rec:
                p = Path(rec['path'])
                records.append((p if p.is_absolute() else ROOT / p, rec['sha256']))
    return records

def plan():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    protected = {key(ROOT / p) for p in tracked if p}
    protected.update(key(p) for p, _ in evidence())
    selection = json.loads((ROOT / 'configs/splatting/selected_models.json').read_text())
    for ds in selection['datasets'].values():
        protected.add(key(ROOT / ds['presentation_checkpoint']))
        run = ROOT / ds['fresh_trained_model'] / 'nerfstudio_models'
        protected.update(key(p) for p in run.glob('*.ckpt'))
    # Preserve the accepted fresh scene control and its final continuation.
    for rel in [
        'outputs/splat_surroundings_tests/train/scene_control',
        'outputs/scene_control_continuation/train/scene_15000',
    ]:
        protected.update(key(p) for p in (ROOT / rel).rglob('*.ckpt'))
    for p in ART.rglob('*.ckpt'):
        if '/models/balanced90/' in p.as_posix():
            protected.add(key(p))
    rows = json.loads((OUT / 'files.json').read_text())
    candidates = []
    for root, rel, size in rows:
        p = Path(root) / rel
        if key(p) in protected or not p.exists():
            continue
        parts = Path(rel).parts
        reason = None
        if Path(root) in (ROOT, ART) and p.suffix == '.ckpt':
            reason = 'Rejected experiment or superseded training checkpoint'
        elif Path(root) == ROOT and rel.startswith('outputs\\') and p.name.startswith('events.out.tfevents.'):
            reason = 'Bulky training event stream; configuration and result reports retained'
        elif Path(root) == ART and parts[:3] == ('gnt_finalization', 'assets', 'interactive_models') and p.suffix == '.bin':
            reason = 'Removed outside-orbit viewer binary'
        elif Path(root) == ROOT and parts[:2] == ('web_demo', 'assets') and len(parts) == 3 and p.suffix == '.bin':
            reason = 'Obsolete live viewer binary; final demo uses route videos'
        elif Path(root) == ART and parts[:3] == ('gnt_finalization', 'assets', 'scene15_splat') and p.name.startswith('splat_') and p.suffix == '.png':
            reason = 'Intermediate full-route splat frame; encoded comparisons retained'
        elif Path(root) == TEMP and parts and ('qa' in parts[0].lower()) and len(parts) > 1:
            reason = 'Isolated project browser verification profile/cache'
        elif Path(root) == TEMP and parts[:2] == ('models', '2dgs') and p.suffix == '.ply':
            reason = 'Rejected 2DGS trial model; trial reports and renders retained'
        elif Path(root) == ROOT and (parts[0] in ('.pytest_cache', '.matplotlib', '.matplotlib-boot2', '.matplotlib-milk2') or '__pycache__' in parts):
            reason = 'Regenerable Python/test/plot cache'
        if reason:
            safe(p)
            s = p.stat()
            candidates.append(dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns, reason=reason))
    # Fingerprint videos and accepted model weights, plus all indexed evidence.
    verification = {key(p): dict(path=str(p), sha256=h) for p, h in evidence()}
    for root, rel, _ in rows:
        p = Path(root) / rel
        if Path(root) in (ROOT, ART) and p.exists() and (p.suffix in ('.mp4', '.pth') or key(p) in protected and p.suffix == '.ckpt'):
            verification[key(p)] = dict(path=str(p), sha256=digest(p))
    for item in verification.values():
        if not Path(item['path']).exists() or digest(item['path']) != item['sha256']:
            raise ValueError(f'Pre-existing evidence mismatch: {item["path"]}')
    doc = dict(created_utc=datetime.now(timezone.utc).isoformat(), allowed_roots=list(map(str, ALLOWED)),
               free_before={d: shutil.disk_usage(d + '\\').free for d in ('C:', 'E:')},
               files=candidates, retained_checks=list(verification.values()))
    (OUT / 'deletion_plan.json').write_text(json.dumps(doc, indent=2))
    totals = collections.Counter()
    for c in candidates:
        totals[c['reason']] += c['bytes']
    print(json.dumps(dict(files=len(candidates), logical_gb=sum(totals.values())/1e9,
                          categories_gb={k: round(v/1e9, 3) for k,v in totals.items()}, retained_checks=len(verification)), indent=2))

def execute():
    doc = json.loads((OUT / 'deletion_plan.json').read_text())
    removed, failed = [], []
    for item in doc['files']:
        p = Path(item['path'])
        try:
            safe(p)
            s = p.stat()
            if s.st_size != item['bytes'] or s.st_mtime_ns != item['mtime_ns']:
                raise ValueError('File changed after plan')
            p.unlink()
            removed.append(item)
        except (OSError, ValueError) as e:
            failed.append(dict(path=str(p), error=str(e)))
    mismatches = []
    for v in doc['retained_checks']:
        p = Path(v['path'])
        if not p.exists() or digest(p) != v['sha256']:
            mismatches.append(str(p))
    after = {d: shutil.disk_usage(d + '\\').free for d in ('C:', 'E:')}
    report = dict(completed_utc=datetime.now(timezone.utc).isoformat(), deleted_count=len(removed),
                  deleted_logical_bytes=sum(v['bytes'] for v in removed), free_before=doc['free_before'], free_after=after,
                  actual_reclaimed_bytes={d: after[d]-doc['free_before'][d] for d in after},
                  retained_checks=len(doc['retained_checks']), evidence_mismatches=mismatches,
                  deleted=removed, failures=failed)
    (OUT / 'cleanup_results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('deleted', 'failures')}, indent=2))
    print('Failures:', len(failed))
    if mismatches:
        raise RuntimeError('Retained evidence verification failed')

if __name__ == '__main__':
    execute() if '--execute' in sys.argv else plan()
