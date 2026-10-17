"""Snapshot documentary evidence for the full project retrospective."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
EXCLUDE={'PROJECT_FULL_WRITEUP.md','PROJECT_EVIDENCE_APPENDIX.md'}
ARTIFACT=Path.home()/'sfm_reconstruction_artifacts/gnt_finalization'


def entry(path):
    data=path.read_bytes()
    try:name=path.relative_to(ROOT).as_posix()
    except ValueError:name=str(path)
    return dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main():
    reports=[p for p in sorted(DOCS.glob('*.md')) if p.name not in EXCLUDE]
    reports += [ROOT/'README.md',ROOT/'web_demo/README.md']
    reports += [p for p in [ROOT/'handoff.md',ROOT/'outputs/splat_closeups/RESULTS.md',ROOT/'outputs/splat_closeups/NATIVE_RESULTS.md',ROOT/'outputs/splat_detail_review/RESULTS.md'] if p.exists()]
    configs=sorted((ROOT/'configs').rglob('*.json'))
    machines=sorted((ROOT/'outputs/gnt_finalization').glob('*.json'))
    machines += [ROOT/p for p in ['outputs/hyperparameter_check/parameter_audit.json','outputs/hyperparameter_check/summary.json','outputs/splatting_cpp/results.json','outputs/splatting_cpp/gpu_summary.json','outputs/splat_refresh_v1/final_selection.json','outputs/splat_surroundings_tests/comparisons.json','outputs/scene_control_continuation/comparisons.json','outputs/depth_research/results.json','outputs/lightweight_methods/results.json','outputs/gnt_tuning/results.json','outputs/gnt_balanced/results.json','outputs/gnt_detail/results.json','outputs/gnt_traversal/results.json'] if (ROOT/p).exists()]
    machines += [ARTIFACT/p for p in ['balanced_rollout/job.json','assets/scene15/job.json','assets/scene15/exports.json','assets/scene15_splat/job.json','assets/scene15_splat/results.json','assets/interactive_models/manifest.json'] if (ARTIFACT/p).exists()]
    for name in ('box','milk','boot','scene'):
        for file in ('balanced90_evaluation.json','video.json'):
            path=ARTIFACT/'assets/balanced_rollout'/name/file
            if path.exists():machines.append(path)
    for folder in ['stage3_gaussian_colmap_stride16_half_global_balanced','stage3_gaussian_submaps_stride16_half','stage3_gaussian_spatial_submaps_stride16_half','stage3_splat_temporal_diagnostics','stage3_splat_temporal_diagnostics_global_balanced']:
        machines+=sorted((ROOT/'outputs'/folder).glob('*.json'))
    machines=list(dict.fromkeys(machines))
    inventory=[]
    for folder in ('src','scripts','native','configs','tests'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.py','.ps1','.cpp','.hpp','.h','.json','.toml','.yml','.yaml','.txt'} and not any(x in path.parts for x in ('__pycache__','vendor','build','python_deps')):
                inventory.append(entry(path))
    for pattern in ('*.html','*.js','*.css','*.mjs'):
        inventory.extend(entry(p) for p in sorted((ROOT/'web_demo').glob(pattern)))
    generated=datetime.now(timezone.utc).isoformat()
    manifest=dict(generated_utc=generated,narrative=entry(DOCS/'PROJECT_FULL_WRITEUP.md'),reports=[entry(p) for p in reports],configurations=[entry(p) for p in configs],machine_records=[entry(p) for p in machines],implementation_inventory=inventory,
        note='Snapshot of available evidence, not proof that every historical command was rerun. Large machine records are indexed by path/hash, not duplicated in full. Historical report statuses remain historical. Deleted/unrecorded files cannot be reconstructed from this snapshot.')
    chunks=['# Complete project evidence appendix\n',f'Generated {generated}.\n',
        'Companion to [the chronological narrative](PROJECT_FULL_WRITEUP.md). This appendix preserves the available documentary record, including failed experiments, complete parameter tables and reproduction commands. The original texts below are historical snapshots: their use of “current,” “not implemented,” or “pending” is not automatically the final project status. Read the narrative for superseding decisions.\n',
        'Source content is copied without editorial correction. Source hashes identify the original files; headings/links inside a copied report retain its original context. Original datasets, images, checkpoints, full per-frame tables and videos remain in their recorded locations rather than being embedded as enormous binary/text dumps. Machine-record paths and SHA-256 values make those limits explicit.\n',
        '## Documentary index\n']
    for i,path in enumerate(reports,1):chunks.append(f'- [{path.name}](#evidence-{i:02d}) - `{entry(path)["path"]}`\n')
    chunks.append('\n## Full original reports\n')
    for i,path in enumerate(reports,1):
        meta=entry(path)
        chunks.extend([f'\n<a id="evidence-{i:02d}"></a>\n\n## Evidence {i:02d}: {path.name}\n\n',f'Source: `{meta["path"]}`; SHA-256: `{meta["sha256"]}`.\n\n---\n\n',path.read_text(encoding='utf-8-sig').rstrip()+'\n\n---\n'])
    chunks.append('\n## Complete configuration snapshots\n\nThese preserve the current recipe/selection files. They do not imply that every historical presentation used the newest geometry.\n')
    for path in configs:
        meta=entry(path);chunks.extend([f'\n### {meta["path"]}\n\nSHA-256: `{meta["sha256"]}`\n\n```json\n',path.read_text(encoding='utf-8-sig').rstrip(),'\n```\n'])
    chunks.append('\n## Machine-readable evidence index\n\nRecords at most 16 KiB are included in full below. Larger records retain their source path, size and hash; the full original file remains the authoritative per-frame/per-run table.\n\n| Record | Bytes | SHA-256 |\n|---|---:|---|\n')
    for path in machines:
        meta=entry(path);chunks.append(f'| `{meta["path"]}` | {meta["bytes"]} | `{meta["sha256"]}` |\n')
    for path in machines:
        if path.stat().st_size<=16384:
            chunks.extend([f'\n### Machine record: {entry(path)["path"]}\n\n```json\n',path.read_text(encoding='utf-8-sig').rstrip(),'\n```\n'])
    chunks.append('\n## Implementation and verification inventory\n\nThis is a file/hash inventory, not a claim that every source file was newly authored or every test was rerun for this retrospective. The machine-readable version is in `PROJECT_EVIDENCE_MANIFEST.json`.\n\n| File | Bytes | SHA-256 |\n|---|---:|---|\n')
    for meta in inventory:chunks.append(f'| `{meta["path"]}` | {meta["bytes"]} | `{meta["sha256"]}` |\n')
    output=DOCS/'PROJECT_EVIDENCE_APPENDIX.md';output.write_text(''.join(chunks),encoding='utf-8')
    manifest['appendix']=entry(output)
    (DOCS/'PROJECT_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(reports=len(reports),configurations=len(configs),machine_records=len(machines),implementation_files=len(inventory),appendix_bytes=output.stat().st_size,narrative_words=len((DOCS/'PROJECT_FULL_WRITEUP.md').read_text(encoding='utf-8').split()))))


if __name__=='__main__':main()
