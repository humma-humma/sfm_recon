"""Render the actual updated web shader with GPU SH textures against reviewed images."""
from pathlib import Path
import json
import re
from time import perf_counter
import moderngl
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/web_renderer_validation'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    text=(ROOT/'web_demo/app.js').read_text(encoding='utf-8')
    def source(kind):
        s=re.search(r'const '+kind+r'=`(.*?)`;',text,re.S).group(1)
        return re.sub(r'precision highp (?:float|int);','',s.replace('#version 300 es','#version 330'))
    ctx=moderngl.create_standalone_context(require=330)
    program=ctx.program(vertex_shader=source('vertex'),fragment_shader=source('fragment'))
    framebuffer=ctx.simple_framebuffer((960,540),components=4);framebuffer.use();ctx.viewport=(0,0,960,540)
    ctx.enable(moderngl.BLEND);ctx.blend_func=moderngl.ONE,moderngl.ONE_MINUS_SRC_ALPHA
    manifest=json.loads((ROOT/'web_demo/assets/manifest.json').read_text())
    report=dict(renderer=ctx.info['GL_RENDERER'],method='Actual updated app shader, offscreen desktop GLSL adaptation. Uploaded float16 SH; comparison to reviewed float32 full-SH projection-guard images. Browser UI is not tested.',datasets={})
    for name,item in manifest.items():
        records=np.fromfile(ROOT/f'web_demo/assets/{name}.bin',dtype='<f4').reshape(-1,16)
        coefficients=np.fromfile(ROOT/f'web_demo/assets/{name}.sh.bin',dtype='<f2').reshape(-1,16,4)
        textures=[]
        for unit,data,dtype in [(0,records.reshape(-1,4),'f4'),(1,coefficients.reshape(-1,4),'f2')]:
            height=int(np.ceil(len(data)/2048));padded=np.zeros((height*2048,4),dtype=data.dtype);padded[:len(data)]=data
            tex=ctx.texture((2048,height),4,padded.tobytes(),dtype=dtype);tex.filter=(moderngl.NEAREST,moderngl.NEAREST);tex.use(unit);textures.append(tex)
        program['splats'].value=0;program['harmonics'].value=1
        indices=ctx.buffer(reserve=len(records)*4);vao=ctx.vertex_array(program,[(indices,'1u /i','splatIndex')])
        cases=json.loads((ROOT/f'outputs/splat_closeups/{name}/camera_cases.json').read_text());rows=[]
        for case in cases:
            eye=np.array(case['position']);f=np.array(case['forward']);r=np.cross(f,case['up']);r/=np.linalg.norm(r);u=np.cross(r,f)
            order=np.argsort(-((records[:,:3]-eye)@f),kind='stable').astype('u4');indices.write(order.tobytes())
            for key,value in dict(eye=eye,forward=f,right=r,up=u,viewport=(960,540)).items():program[key].value=tuple(value)
            program['focal'].value=540/(2*np.tan(np.radians(case['fov']/2)))
            samples=[]
            for repeat in range(3):
                framebuffer.clear(.035,.045,.035,1);ctx.finish();tick=perf_counter();vao.render(mode=moderngl.TRIANGLE_STRIP,vertices=4,instances=len(records));ctx.finish();samples.append(1000*(perf_counter()-tick))
            image=Image.frombytes('RGB',(960,540),framebuffer.read(components=3)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            reference=np.asarray(Image.open(ROOT/f'outputs/splat_closeups/{name}/{case["id"]}_full_sh_guard.png').convert('RGB'),dtype=float)
            delta=np.asarray(image,dtype=float)-reference;mse=float(np.mean(delta*delta));mae=float(np.abs(delta).mean());psnr=float(10*np.log10(255**2/max(mse,1e-12)))
            assert psnr>=45 and mae<.2,(name,case['id'],psnr,mae)
            file=f'{name}_{case["id"]}.png';image.save(OUT/file)
            rows.append(dict(case=case['id'],file=file,reference_psnr=psnr,mean_abs_rgb_0_255=mae,gpu_draw_ms=float(np.median(samples[1:]))))
        report['datasets'][name]=dict(tests=len(rows),min_psnr=min(r['reference_psnr'] for r in rows),max_rgb_mae=max(r['mean_abs_rgb_0_255'] for r in rows),median_gpu_draw_ms=float(np.median([r['gpu_draw_ms'] for r in rows])),cases=rows)
        print(name,json.dumps({k:v for k,v in report['datasets'][name].items() if k!='cases'}),flush=True)
        for tex in textures:tex.release()
        vao.release();indices.release()
    report['status']='pass';report['tests']=sum(d['tests'] for d in report['datasets'].values())
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print('PASS: updated GPU shader matches reviewed candidate in all 72 cases.',flush=True)

if __name__=='__main__':main()
