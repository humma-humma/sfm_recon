"""Audit selected models' saved resolution schedules without loading/training models."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/splat_surroundings_audit'
COLMAP={
    'box':'outputs/splat_refresh_v1/colmap/box',
    'boot':'outputs/stage2_boot_gaussian_colmap_half',
    'milk':'outputs/stage2_milk_gaussian_colmap_half',
    'scene':'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced',
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected=json.loads((ROOT/'configs/splatting/selected_models.json').read_text())['datasets']
    report={'method':'Saved config + COLMAP input dimensions; Nerfstudio 1.1.5 training factor = 2**max(num_downscales - step//resolution_schedule, 0). Evaluation factor = 1. No training was run or models edited.',
        'source':'https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/models/splatfacto.py', 'datasets':{}}
    lines=['# Surroundings: training resolution audit','',report['method'],'',
        '| Dataset | Iterations | Training resolutions and step ranges | Full-input-resolution steps | Splitting stops |',
        '|---|---:|---|---:|---:|']
    for name,choice in selected.items():
        text=(ROOT/choice['presentation_config']).read_text()
        def number(key):return int(re.search(r'^\s*'+key+r': (\d+)\s*$',text,re.M).group(1))
        iterations=number('max_num_iterations');schedule=number('resolution_schedule');downscales=number('num_downscales');stop=number('stop_split_at')
        camera=next(line.split() for line in (ROOT/COLMAP[name]/'sparse/0/cameras.txt').read_text().splitlines() if line and not line.startswith('#'))
        width,height=map(int,camera[2:4]);stages=[]
        for start in range(0,iterations,schedule):
            end=min(iterations,start+schedule);factor=2**max(downscales-start//schedule,0)
            if stages and stages[-1]['downscale_factor']==factor:
                stages[-1]['end_exclusive']=end
            else:
                stages.append(dict(start=start,end_exclusive=end,downscale_factor=factor,resolution=[width//factor,height//factor]))
        full=sum(stage['end_exclusive']-stage['start'] for stage in stages if stage['downscale_factor']==1)
        split_resolution_factor=2**max(downscales-max(0,stop-1)//schedule,0)
        report['datasets'][name]=dict(config=choice['presentation_config'],iterations=iterations,input_resolution=[width,height],num_downscales=downscales,resolution_schedule=schedule,stages=stages,full_resolution_training_steps=full,stop_split_at=stop,resolution_before_split_stop=[width//split_resolution_factor,height//split_resolution_factor])
        display='; '.join(f'{s["resolution"][0]}x{s["resolution"][1]} @ {s["start"]}-{s["end_exclusive"]-1}' for s in stages[:downscales+1])
        lines.append(f'| {name} | {iterations} | {display} | {full} | {stop} |')
    lines+=['','## Interpretation','',
        'The selected scene model never trains at its 960x540 input resolution. Gaussian splitting stops while training is still at 240x135. This is a concrete configuration limitation and a plausible contributor to missing lettering and environmental texture, not proof that it is the only cause. Boot/milk train at 960x540 from step 6000 onward, so the same schedule mismatch does not explain their remaining blur.','',
        'First isolated scene experiment: keep the same input images, poses, seed, 5000 steps and all other settings; change resolution_schedule from 3000 to 1000. This reaches 960x540 at step 2000. Compare captured-view environment crops, all held-out views, close-ups and GPU memory/runtime. Do not automatically replace the selected model.','',
        'Second separate experiment, if needed: extend stop_split_at from 2500 to 4000 while keeping the faster resolution schedule, to allow more refinement at the final resolution. Original 1920x1080 inputs are a further separate experiment, with higher memory cost.','',
        'For object surroundings, the source frames include changing people/occlusions and reflective windows/appliances. Masking moving distractors or capturing a static room are candidates. A cropped object/studio view cleans presentation but does not reconstruct the room more accurately. Aggressive depth cleanup previously damaged detail and should not be enabled as the default fix.','',
        '[Nerfstudio 1.1.5 resolution schedule implementation]('+report['source']+')']
    (OUT/'report.json').write_text(json.dumps(report,indent=2));(OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
