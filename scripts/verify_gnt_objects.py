"""Verify completed object archives and gallery resources without GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import cv2

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    reports=json.loads((ROOT/'outputs/gnt_finalization/objects.json').read_text(encoding='utf-8'))
    if not args.partial:assert set(reports)=={'box','boot','milk'}
    checked=0;models=0
    for name,report in reports.items():
        assert report['status']=='complete'
        assert set(report['checkpoints'])=={'600','1800','6000'}
        assert all(i%8 for group in report['groups'] for i in group)
        for model in report['checkpoints'].values():
            path=args.destination/'models'/name/model['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==model['sha256'];models+=1
        keys=[(r['case'],r['label'],r['width']) for r in report['runs']]
        assert len(keys)==len(set(keys))
        for row in report['runs']:
            assert all(i%8 for i in row['sources'])
            path=ROOT/'web_demo'/row['archive_url']
            image=cv2.imread(str(path));assert image.shape[:2]==(row['height'],row['width'])
            with urllib.request.urlopen(urllib.request.Request('http://localhost:8088/'+row['archive_url'],method='HEAD')) as response:
                assert response.status==200
            checked+=1
        for case in report['review_cases']:
            width=480 if case['kind']=='motion' else 960
            assert all((case['id'],label,width) in keys for label in ('pretrained','balanced1800','balanced6000'))
    page=(ROOT/'web_demo/gnt-objects.html').read_text(encoding='utf-8')
    data=json.loads(re.search(r'<script id="comparison-data" type="application/json">(.*?)</script>',page,re.S)[1])
    for cases in data['cases'].values():
        for case in cases:
            assert all(label in case['urls'] for label in ('pretrained','balanced1800','balanced6000'))
            if case['kind']=='motion':assert 'gt' not in case['urls']
            else:assert 'gt' in case['urls']
            for url in case['urls'].values():
                image=cv2.imread(str(ROOT/'web_demo'/url));assert image.shape[:2]==(case['height'],case['width'])
    proof=dict(status='passed',datasets=list(reports),models=models,archived_images=checked,
        heldout_exclusion=True,gallery_dimensions=True,complete=not args.partial)
    (ROOT/'outputs/gnt_finalization/object_verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
