"""Read-only offscreen GPU diagnostic of the web demo shader, not Nerfstudio.

Produces all-four-dataset close-up comparisons without changing models or demo.
Requires the local moderngl/torch/numpy/Pillow environment. The full-SH test
changes color only; projection and oversized-splat tests change geometry handling.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from time import perf_counter
import moderngl
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/splat_closeups'
WIDTH,HEIGHT=960,540
VARIANTS=('browser','near_only','clamp_only','projection_guard','full_sh','full_sh_guard','large_splat_clip')
LABELS={'browser':'Current demo / SH0','near_only':'Near clip only: .005 to .01','clamp_only':'Covariance projection clamp only','projection_guard':'Near clip + projection clamp / SH0','full_sh':'Current projection / full SH','full_sh_guard':'Near clip + projection clamp / full SH','large_splat_clip':'Diagnostic: clip large splats'}

def unit(v): return v/max(np.linalg.norm(v),1e-12)

def sh_colors(coefficients,means,eye):
    # Orthonormal real spherical harmonics through degree 3 in standard 3DGS order.
    directions=means-eye
    directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
    x,y,z=directions.T;xx=x*x;yy=y*y;zz=z*z
    basis=[np.full(len(x),1/np.sqrt(4*np.pi)),
           -np.sqrt(3/(4*np.pi))*y,np.sqrt(3/(4*np.pi))*z,-np.sqrt(3/(4*np.pi))*x,
           np.sqrt(15/(4*np.pi))*x*y,-np.sqrt(15/(4*np.pi))*y*z,
           np.sqrt(5/(16*np.pi))*(2*zz-xx-yy),-np.sqrt(15/(4*np.pi))*x*z,
           np.sqrt(15/(16*np.pi))*(xx-yy),
           -np.sqrt(35/(32*np.pi))*y*(3*xx-yy),np.sqrt(105/(4*np.pi))*x*y*z,
           -np.sqrt(21/(32*np.pi))*y*(4*zz-xx-yy),np.sqrt(7/(16*np.pi))*z*(2*zz-3*xx-3*yy),
           -np.sqrt(21/(32*np.pi))*x*(4*zz-xx-yy),np.sqrt(105/(16*np.pi))*z*(xx-yy),
           -np.sqrt(35/(32*np.pi))*x*(xx-3*yy)]
    rgb=np.full((len(x),3),.5,dtype=np.float32)
    for i,b in enumerate(basis[:coefficients.shape[1]]):rgb+=b[:,None]*coefficients[:,i]
    return np.maximum(rgb,0)

def shaders():
    source=(ROOT/'web_demo/app.js').read_text(encoding='utf-8')
    # Freeze the original SH0 demo so updating the app cannot change this experiment.
    vertex=(ROOT/'scripts/fixtures/splat_closeup_baseline.vert').read_text(encoding='utf-8')
    fragment=re.search(r'const fragment=`(.*?)`;',source,re.S).group(1)
    def desktop(s):
        s=s.replace('#version 300 es','#version 330')
        s=re.sub(r'precision highp (?:float|int);','',s)
        return s.replace('gl_VertexID','gl_VertexID')
    vertex=desktop(vertex);fragment=desktop(fragment)
    guarded=vertex.replace('uniform sampler2D splats;', 'uniform int variant; uniform sampler2D splats;')
    guarded=guarded.replace('if(v.z<0.005)', 'if(v.z<((variant==1||variant==4)?0.01:0.005))')
    guarded=guarded.replace('vec3 jx=focal/v.z*(right-forward*v.x/v.z),jy=focal/v.z*(up-forward*v.y/v.z);',
        'vec2 slope=v.xy/v.z; if(variant==1||variant==2) slope=clamp(slope,-1.3*viewport/(2.*focal),1.3*viewport/(2.*focal));\n'
        'vec3 jx=focal/v.z*(right-forward*slope.x),jy=focal/v.z*(up-forward*slope.y);')
    guarded=guarded.replace('vec2 corner=vec2(',
        'if(variant==3 && sqrt(max(.1,mid+gap))>viewport.y*.2){gl_Position=vec4(2.,2.,2.,1.);local=vec2(9.);tint=vec4(0.);return;}\nvec2 corner=vec2(')
    return vertex,guarded,fragment

def target_distance(data,view):
    eye=np.array(view['position']);f=unit(np.array(view['forward']));u=unit(np.array(view['up']));r=unit(np.cross(f,u))
    delta=data[:,:3]-eye;z=delta@f;x=delta@r;y=delta@u
    # Estimate visible central surface depth, ignoring almost-transparent records.
    valid=(z>.01)&(np.abs(x)<.12*z)&(np.abs(y)<.12*z)&(data[:,3]>.1)
    if valid.sum()<10:valid=(z>.01)&(np.abs(x)<.4*z)&(np.abs(y)<.4*z)
    if not valid.any():raise ValueError('Cannot determine central depth')
    return float(np.median(z[valid]))

def cases(item,data):
    views=item['views'];indices=sorted({0,((len(views)-1)//16)*8,((len(views)-1)//8)*8})
    result=[]
    for index in indices:
        view=views[index];distance=target_distance(data,view)
        for label,fraction,zoom in [('captured',0,1),('dolly25',.25,1),('dolly50',.5,1),('dolly75',.75,1),('optical2x',0,2),('optical4x',0,4)]:
            eye=np.array(view['position'])+np.array(view['forward'])*distance*fraction
            fov=float(np.degrees(2*np.arctan(np.tan(np.radians(view['fov']/2))/zoom)))
            result.append(dict(id=f'view{index:03d}_{label}',camera_index=index,source_image=view['name'],case=label,
                position=eye.tolist(),forward=view['forward'],up=view['up'],fov=fov,estimated_center_depth=distance,dolly_fraction=fraction,zoom=zoom))
    return result

def tile(image,label,width=400):
    image=image.convert('RGB').resize((width,round(image.height*width/image.width)))
    out=Image.new('RGB',(width,image.height+30),'#172019');out.paste(image,(0,30));ImageDraw.Draw(out).text((8,8),label,fill='white');return out

def gallery(report):
    html=['<!doctype html><meta charset="utf-8"><title>Splat close-up comparisons</title><style>body{background:#101510;color:#e4e9df;font:14px Segoe UI,sans-serif;margin:24px}h1{font-weight:500}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}img{width:100%;display:block}figure{margin:0}figcaption{padding:8px;background:#1d281e}a{color:#bfdc9b}section{margin:30px 0}nav{display:flex;gap:20px;position:sticky;top:0;background:#101510;padding:14px}p{max-width:1100px;line-height:1.6}</style>',
        '<h1>Close-up render comparisons</h1><p>Read-only diagnostic. Same selected checkpoints and cameras in every column. Columns isolate the current browser shader, projection safeguards, full spherical-harmonic color, and aggressive large-splat clipping. Clipping is diagnostic and may remove real surfaces. These are actual RTX 2080 offscreen OpenGL renders of the demo shader, not browser screenshots or fresh Nerfstudio renders. No models or production demo settings were changed.</p>',
        '<p>Captured rows include saved source/Nerfstudio pairs below. Novel close-up Nerfstudio rendering remains blocked by Docker access. Dolly moves the camera toward estimated central depth; optical zoom holds the camera fixed. The depth estimate is heuristic, not ground truth. Click any image to inspect at full resolution.</p>',
        '<nav>'+''.join(f'<a href="#{n}">{n.title()}</a>' for n in report['datasets'])+'</nav>']
    for name,dataset in report['datasets'].items():
        html.append(f'<h2 id="{name}">{name.title()}</h2><p>{dataset["gaussians"]:,} Gaussians</p>')
        groups={}
        for row in dataset['renders']:groups.setdefault(row['case_id'],[]).append(row)
        for case_id,rows in groups.items():
            html.append(f'<section><h3>{case_id} · source {rows[0]["source_image"]}</h3><div class="grid">')
            rows.sort(key=lambda row:VARIANTS.index(row['variant']))
            for row in rows:
                file=row['file'];html.append(f'<figure><a href="{file}"><img loading="lazy" src="{file}" alt="{name} {case_id} {row["variant"]}"></a><figcaption>{LABELS[row["variant"]]} · {row["gpu_render_ms"]:.1f} ms</figcaption></figure>')
            html.append('</div>')
            reference=dataset['references'].get(str(rows[0]['camera_index']))
            if reference and rows[0]['case']=='captured':html.append(f'<p>Saved source image / Nerfstudio render (same captured camera):</p><a href="{reference}"><img style="max-width:960px" loading="lazy" src="{reference}"></a>')
            html.append('</section>')
        # Compact first-camera overview, all six zoom/movement levels.
        first=min(r['camera_index'] for r in dataset['renders']);panels=[]
        for case_id,rows in groups.items():
            if rows[0]['camera_index']!=first:continue
            tiles=[tile(Image.open(OUT/r['file']),r['case']+' / '+r['variant']) for r in rows if r['variant'] in ('browser','projection_guard','full_sh_guard','large_splat_clip')]
            panel=Image.new('RGB',(sum(i.width for i in tiles),tiles[0].height))
            x=0
            for t in tiles:panel.paste(t,(x,0));x+=t.width
            panels.append(panel)
        sheet=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)))
        y=0
        for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
        sheet.save(OUT/f'{name}_overview.jpg',quality=94)
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',default=['box','boot','milk','scene']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'web_demo/assets/manifest.json').read_text())
    ctx=moderngl.create_standalone_context(require=330)
    original,guarded,fragment=shaders()
    programs={variant:ctx.program(vertex_shader=original if variant in ('browser','full_sh') else guarded,fragment_shader=fragment) for variant in VARIANTS}
    framebuffer=ctx.simple_framebuffer((WIDTH,HEIGHT),components=4);framebuffer.use();ctx.viewport=(0,0,WIDTH,HEIGHT)
    ctx.enable(moderngl.BLEND);ctx.blend_func=moderngl.ONE,moderngl.ONE_MINUS_SRC_ALPHA
    report=dict(status='running',renderer=ctx.info['GL_RENDERER'],resolution=[WIDTH,HEIGHT],browser_test='blocked: no connected browser',nerfstudio_closeups='blocked: Docker Linux engine Access is denied',method='Same demo shader adapted from GLSL ES 300 to desktop GLSL 330; current models unchanged; GPU timings exclude sorting/color evaluation/upload/readback',datasets={})
    start=perf_counter()
    for name in args.datasets:
        item=manifest[name];data=np.fromfile(ROOT/f'web_demo/assets/{name}.bin',dtype='<f4').reshape(-1,16)
        state=torch.load(ROOT/item['checkpoint'],map_location='cpu',weights_only=False)['pipeline']
        coeff=np.concatenate([state['_model.gauss_params.features_dc'].numpy()[:,None,:],state['_model.gauss_params.features_rest'].numpy()],axis=1)
        if len(coeff)!=len(data):raise ValueError('Exporter filtered records; SH alignment needs matching mask')
        del state
        matrix=cases(item,data);folder=OUT/name;folder.mkdir(exist_ok=True)
        (folder/'camera_cases.json').write_text(json.dumps(matrix,indent=2))
        dataset=dict(checkpoint=item['checkpoint'],gaussians=len(data),renders=[],references={})
        report['datasets'][name]=dataset
        eval_dir=ROOT/'outputs/splat_refresh_v1/evaluation'/('box_new' if name=='box' else name+'_old')
        evaluation=json.loads((eval_dir/'metrics.json').read_text())
        for camera_index in sorted({c['camera_index'] for c in matrix}):
            image=item['views'][camera_index]['name'];reference_index=next((i for i,v in enumerate(evaluation['views']) if v['image']==image),None)
            if reference_index is not None:
                reference=Image.open(eval_dir/f'{reference_index:03d}_img.png');file=f'{name}/view{camera_index:03d}_source_nerfstudio.png';reference.save(OUT/file);dataset['references'][str(camera_index)]=file
        height=int(np.ceil(len(data)*4/2048));padded=np.zeros((height*2048,4),dtype='f4')
        texture=ctx.texture((2048,height),4,dtype='f4');texture.filter=(moderngl.NEAREST,moderngl.NEAREST);texture.use(0)
        index_buffer=ctx.buffer(reserve=len(data)*4)
        arrays={v:ctx.vertex_array(programs[v],[(index_buffer,'1u /i','splatIndex')]) for v in VARIANTS}
        for case in matrix:
            eye=np.array(case['position']);forward=unit(np.array(case['forward']));right=unit(np.cross(forward,case['up']));up=unit(np.cross(right,forward))
            tick=perf_counter();depth=(data[:,:3]-eye)@forward;order=np.argsort(-depth,kind='stable').astype('u4');sort_ms=1000*(perf_counter()-tick);index_buffer.write(order.tobytes())
            full_color=sh_colors(coeff,data[:,:3],eye)
            for variant in VARIANTS:
                records=data.copy()
                if variant in ('full_sh','full_sh_guard'):records[:,4:7]=full_color
                padded[:len(data)*4]=records.reshape(-1,4);texture.write(padded.tobytes())
                program=programs[variant]
                for key,value in dict(eye=eye,right=right,up=up,forward=forward,viewport=(WIDTH,HEIGHT)).items():program[key].value=tuple(value)
                program['focal'].value=HEIGHT/(2*np.tan(np.radians(case['fov']/2)));program['splats'].value=0
                if 'variant' in program:program['variant'].value={'projection_guard':1,'full_sh_guard':1,'clamp_only':2,'large_splat_clip':3,'near_only':4}[variant]
                samples=[]
                for repeat in range(3):
                    framebuffer.clear(.035,.045,.035,0);ctx.finish();tick=perf_counter();arrays[variant].render(mode=moderngl.TRIANGLE_STRIP,vertices=4,instances=len(data));ctx.finish();samples.append(1000*(perf_counter()-tick))
                image=Image.frombytes('RGBA',(WIDTH,HEIGHT),framebuffer.read(components=4)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert('RGB')
                file=f'{name}/{case["id"]}_{variant}.png';image.save(OUT/file)
                dataset['renders'].append(dict(case_id=case['id'],case=case['case'],camera_index=case['camera_index'],source_image=case['source_image'],variant=variant,file=file,gpu_render_ms=float(np.median(samples[1:])),gpu_samples_ms=samples,sort_ms=sort_ms))
            print(name,case['id'],f'sort {sort_ms:.1f} ms',flush=True)
            (OUT/'report.json').write_text(json.dumps(report,indent=2))
        for vao in arrays.values():vao.release()
        texture.release();index_buffer.release()
        gallery(report)
    report.update(status='local_gpu_matrix_complete_external_comparison_blocked',elapsed_seconds=perf_counter()-start,renders=sum(len(d['renders']) for d in report['datasets'].values()))
    (OUT/'report.json').write_text(json.dumps(report,indent=2));gallery(report)
    print(json.dumps({k:v for k,v in report.items() if k!='datasets'},indent=2),flush=True)

if __name__=='__main__':main()
