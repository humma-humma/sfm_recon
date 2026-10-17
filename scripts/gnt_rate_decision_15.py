"""Add a 15fps anchor test, compared fairly against the existing 10fps videos."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from gnt_rate_decision import encode,evaluate,publish,ROOT
from gnt_rate_sampling import split_frames
from gnt_rollout_io import digest,save
from publish_gnt_videos import verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ffmpeg',default=r'C:\ffmpeg\bin\ffmpeg.exe');args=parser.parse_args()
    source=args.root/'assets/dense_turn';previous=args.root/'assets/rate_decision'
    progress=json.loads((source/'progress.json').read_text());old=json.loads((previous/'results.json').read_text())
    assert progress['status']==old['status']=='complete' and progress['checkpoint_sha256']==old['checkpoint_sha256']
    for i,row in enumerate(progress['frames']):assert row['index']==i and digest(source/row['file'])==row['sha256']
    folder=args.root/'assets/rate_decision_15';folder.mkdir(parents=True,exist_ok=True)
    splits={fps:split_frames(fps) for fps in (10,15)}
    common=sorted(set(splits[10]['withheld'])&set(splits[15]['withheld']));assert len(common)==59
    report=dict(status='running',checkpoint_sha256=progress['checkpoint_sha256'],common_withheld=common,splits=splits,videos={},
                note='Same outdoor clip, fixed FFmpeg settings. Existing 10fps videos reused byte-for-byte. New common subset excludes anchors at both rates; scores differ from the earlier 5/10fps subset. No new GNT or splat inference.')
    for label in ('reference','fps10','fps10_held'):
        original=previous/old['videos'][label]['file'];target=folder/original.name
        if not target.exists():shutil.copy2(original,target)
        assert digest(target)==digest(original)==old['videos'][label]['sha256']
        anchors=list(range(180)) if label=='reference' else splits[10]['anchors']
        row={**verify(target,6.),**evaluate(target,source,anchors,common)}
        if label!='reference':row.update(anchor_fps=10,method='held' if label.endswith('held') else 'motion')
        report['videos'][label]=row
        print(json.dumps(dict(label=label,**row['common_withheld'])),flush=True)
    inputs=folder/'inputs_15';inputs.mkdir(exist_ok=True)
    for j,i in enumerate(splits[15]['anchors']):
        target=inputs/f'anchor_{j:04}.png'
        if not target.exists():shutil.copy2(source/f'frame_{i:04}.png',target)
        assert digest(target)==digest(source/f'frame_{i:04}.png')
    for method in ('motion','held'):
        output=folder/f'fps15_{method}.mp4';filters=encode(args.ffmpeg,inputs,output,15,method)
        row={**verify(output,6.),**evaluate(output,source,splits[15]['anchors'],common),'anchor_fps':15,'method':method,'filter':filters}
        label='fps15' if method=='motion' else 'fps15_held';report['videos'][label]=row
        save(folder/'results.json',report);print(json.dumps(dict(label=label,**row['common_withheld'],minimum_anchor_psnr=row['minimum_anchor_psnr'])),flush=True)
    for label,row in report['videos'].items():
        if label=='reference':continue
        split=splits[row['anchor_fps']]
        row['rate_withheld']=dict(count=len(split['withheld']),psnr=float(np.mean([row['rows'][i]['psnr'] for i in split['withheld']])))
        row['held_tail_mean_psnr']=float(np.mean([row['rows'][i]['psnr'] for i in split['held_tail']]))
    report['status']='complete';save(folder/'results.json',report);save(ROOT/'outputs/gnt_finalization/rate_decision_15.json',report)
    publish(folder,report,rates=(10,15),asset_folder='rate_decision_15',page_name='gnt-rate-decision-15.html')
    print('Complete: http://localhost:8088/gnt-rate-decision-15.html',flush=True)


if __name__=='__main__':main()
