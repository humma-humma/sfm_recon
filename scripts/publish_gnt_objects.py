"""Publish object adaptation incrementally into a separate review gallery."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def archive(original,target):
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest, ('Preserving unexpected archive',target)
    else:
        temporary=target.with_suffix(target.suffix+'.partial')
        shutil.copy2(original,temporary)
        assert hashlib.sha256(temporary.read_bytes()).hexdigest()==digest
        temporary.replace(target)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    source=args.trial/'gnt_finalization';assets=args.destination/'assets/objects';assets.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'outputs/lightweight_methods/results.json').read_text(encoding='utf-8'))
    manifest=json.loads((args.trial.parent/'sfm_sharpness_trials_v1/render_manifest.json').read_text(encoding='utf-8'))
    labels=dict(baseline='Original splats',preferred='Preferred restoration',gt='Held-out photograph',
        pretrained='Pretrained GNT',balanced1800='Object GNT 1800',balanced6000='Object GNT 6000')
    data=dict(labels=labels,cases={},default_left='balanced6000',default_right='pretrained');summaries={}
    for name in ('box','boot','milk'):
        report_path=source/name/'results.json'
        if not report_path.exists():continue
        report=json.loads(report_path.read_text(encoding='utf-8'));summaries[name]=report
        entries={}
        for row in report['runs']:
            cid=row['case'];width=row['width'];height=row['height']
            original=source/name/row['path'];target=assets/f'{name}_{cid}_{row["label"]}_{width}.png'
            archive(original,target)
            image=cv2.imread(str(target));assert image.shape[:2]==(height,width)
            row['archive_url']='assets/gnt_finalization/objects/'+target.name
            if row['width']!=960 and row['kind']!='motion':continue
            if cid not in entries:
                previous=next((c for c in old['datasets'][name]['cases'] if c['id']==cid),None)
                case=next((c for c in report['review_cases'] if c['id']==cid),None)
                entries[cid]=dict(id=cid,image=case['image'],kind=row['kind'],zoom=case.get('zoom',1),
                    common_comparison=True,width=width,height=height,urls={})
                if previous:
                    for key in ('baseline','preferred','gt'):
                        if key not in previous['urls']:continue
                        if width==previous['width']:
                            entries[cid]['urls'][key]=previous['urls'][key]
                        else:
                            image=cv2.imread(str(ROOT/'web_demo'/previous['urls'][key]));assert image is not None
                            target=assets/f'{name}_{cid}_{key}_{width}.png'
                            assert cv2.imwrite(str(target),cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA))
                            entries[cid]['urls'][key]='assets/gnt_finalization/objects/'+target.name
            entries[cid]['urls'][row['label']]=row['archive_url']
        for cid,entry in entries.items():
            if entry['kind']=='heldout' and 'gt' not in entry['urls']:
                record=manifest['datasets'][name]
                case=next(c for c in report['review_cases'] if c['id']==cid)
                image=cv2.imread(str(ROOT/record['data']/'images'/case['image']));assert image is not None
                def intrinsic(k):return np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1.]])
                image=cv2.warpPerspective(image,intrinsic(case['k'])@np.linalg.inv(intrinsic(record['k'])),(960,540))
                target=assets/f'{name}_{cid}_gt.png';assert cv2.imwrite(str(target),image)
                entry['urls']['gt']='assets/gnt_finalization/objects/'+target.name
        if entries:data['cases'][name]=list(entries.values())
        models=args.destination/'models'/name;models.mkdir(parents=True,exist_ok=True)
        for record in report['checkpoints'].values():
            target=models/record['path']
            archive(source/name/record['path'],target)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']
    out=ROOT/'outputs/gnt_finalization';out.mkdir(exist_ok=True)
    (out/'objects.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    (assets/'results.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    status=' | '.join(f'{name}: {summaries.get(name,{}).get("status","pending")}' for name in ('box','boot','milk'))
    page='''<!doctype html><meta charset="utf-8"><title>GNT object adaptation</title>
<style>body{background:#15191e;color:#eee;font:16px system-ui;margin:24px}a{color:#9bd}p{max-width:1100px;line-height:1.5}select,button,input{margin:8px;padding:6px}canvas{display:block;width:100%;max-width:960px;touch-action:none;cursor:grab}</style>
<h1>GNT object adaptation</h1><p>__STATUS__</p><p>Each object starts from official pretrained GNT, with balanced camera coverage and RGB training. Compare both checkpoints with pretrained GNT, the preferred restoration, and matching held-out photos where available. Motion views have no photograph reference. No model is promoted automatically.</p><p><a href="gnt-motion.html">Motion blending tests</a> | <a href="gnt-traversal.html">Accepted scene traversal</a></p>
<label>Dataset <select id="dataset"></select></label><label>View <select id="view"></select></label><br>
<label>Left <select id="left"></select></label><label>Right <select id="right"></select></label><br>
<label>Wipe <input id="wipe" type="range" min="0" max="100" value="50"></label><button id="toggle">Toggle A/B</button>
<label>Crop zoom <input id="zoom" type="range" min="1" max="4" step=".25" value="1"></label><button id="reset">Reset crop</button>
<p id="status"></p><canvas id="comparison" width="960" height="540"></canvas>
<script id="comparison-data" type="application/json">__DATA__</script><script src="gnt-tuning-inspector.js" defer></script>'''
    if not data['cases']:
        page=page.split('<label>Dataset')[0]+'<p>Object training is underway. Refresh after the first object finishes rendering.</p>'
    (ROOT/'web_demo/gnt-objects.html').write_text(page.replace('__STATUS__',status).replace('__DATA__',json.dumps(data)),encoding='utf-8')
    print(json.dumps(dict(status=status,datasets=list(data['cases']))),flush=True)

if __name__=='__main__':main()
