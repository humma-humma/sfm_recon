const $=id=>document.getElementById(id);
const canvas=$('canvas'),status=$('loading');
const options=new URLSearchParams(location.search),assetBase=options.get('models')==='balanced'?'assets/gnt_finalization/interactive_models':'assets',outside=options.get('view')==='outside';
if(options.get('embed')==='1')document.body.classList.add('embedded');
const add=(a,b)=>a.map((v,i)=>v+b[i]), sub=(a,b)=>a.map((v,i)=>v-b[i]),mul=(a,s)=>a.map(v=>v*s),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>mul(a,1/Math.max(1e-12,Math.hypot(...a)));
const rotate=(v,axis,t)=>add(add(mul(v,Math.cos(t)),mul(cross(axis,v),Math.sin(t))),mul(axis,dot(axis,v)*(1-Math.cos(t))));
let gl,program,texture,shTexture,indexBuffer,worker,manifest,current,position,forward,up,center,fov=60,count=0,fly=false,playing=false,tourTime=0,requestId=0,aborter,busy=false,sortDirty=false,lastSort=0,lastTime=0,sequence=0;
const keys=new Set();
window.addEventListener('hashchange',()=>{const name=location.hash.slice(1);if(manifest&&Object.hasOwn(manifest,name)&&current!==name)load(name);});
const vertex=`#version 300 es
precision highp float; precision highp int;
layout(location=0) in uint splatIndex;
uniform sampler2D splats; uniform sampler2D harmonics; uniform vec3 eye,right,up,forward; uniform vec2 viewport; uniform float focal;
out vec2 local; out vec4 tint;
vec4 readAt(int i){return texelFetch(splats,ivec2(i%2048,i/2048),0);}
vec3 shAt(int i){return texelFetch(harmonics,ivec2(i%2048,i/2048),0).rgb;}
vec3 splatColor(int i,vec3 direction){
int h=i*16;float x=direction.x,y=direction.y,z=direction.z;
float xx=x*x,yy=y*y,zz=z*z;
vec3 rgb=vec3(.5)+.28209479177387814*shAt(h);
rgb+=.4886025119029199*(-y*shAt(h+1)+z*shAt(h+2)-x*shAt(h+3));
rgb+=1.0925484305920792*(x*y*shAt(h+4)-y*z*shAt(h+5)-x*z*shAt(h+7));
rgb+=.31539156525252005*(2.*zz-xx-yy)*shAt(h+6)+.5462742152960396*(xx-yy)*shAt(h+8);
rgb+=-.5900435899266435*y*(3.*xx-yy)*shAt(h+9)+2.890611442640554*x*y*z*shAt(h+10);
rgb+=-.4570457994644658*y*(4.*zz-xx-yy)*shAt(h+11)+.3731763325901154*z*(2.*zz-3.*xx-3.*yy)*shAt(h+12);
rgb+=-.4570457994644658*x*(4.*zz-xx-yy)*shAt(h+13)+1.445305721320277*z*(xx-yy)*shAt(h+14);
rgb+=-.5900435899266435*x*(xx-3.*yy)*shAt(h+15);
return max(rgb,vec3(0.));}
void main(){int b=int(splatIndex)*4;vec4 p=readAt(b);vec4 c=readAt(b+1);vec4 a=readAt(b+2);vec4 d=readAt(b+3);
vec3 delta=p.xyz-eye; vec3 v=vec3(dot(delta,right),dot(delta,up),dot(delta,forward));
if(v.z<0.01){gl_Position=vec4(2.,2.,2.,1.);local=vec2(9.);tint=vec4(0.);return;}
mat3 cov=mat3(a.x,a.y,a.z,a.y,a.w,d.x,a.z,d.x,d.y);
// Bound the covariance Jacobian for offscreen splats near the camera plane.
vec2 slope=clamp(v.xy/v.z,-1.3*viewport/(2.*focal),1.3*viewport/(2.*focal));
vec3 jx=focal/v.z*(right-forward*slope.x),jy=focal/v.z*(up-forward*slope.y);
float xx=dot(jx,cov*jx)+.3,xy=dot(jx,cov*jy),yy=dot(jy,cov*jy)+.3;
float mid=.5*(xx+yy),gap=sqrt(max(0.,.25*(xx-yy)*(xx-yy)+xy*xy));
vec2 axis=abs(xy)>1e-7?normalize(vec2(xy,mid+gap-xx)):(xx>=yy?vec2(1.,0.):vec2(0.,1.));
vec2 major=axis*min(sqrt(max(.1,mid+gap)),2048.),minor=vec2(-axis.y,axis.x)*min(sqrt(max(.1,mid-gap)),2048.);
vec2 corner=vec2((gl_VertexID==1||gl_VertexID==3)?1.:-1.,gl_VertexID>=2?1.:-1.);local=corner*3.;
vec2 screen=focal*v.xy/v.z+3.*(major*corner.x+minor*corner.y);
gl_Position=vec4(2.*screen/viewport,0.,1.);tint=vec4(splatColor(int(splatIndex),normalize(delta)),p.w);}`;
const fragment=`#version 300 es
precision highp float;in vec2 local;in vec4 tint;out vec4 color;
void main(){float r=dot(local,local);if(r>9.)discard;float alpha=min(.99,tint.a*exp(-.5*r));if(alpha<.0039)discard;color=vec4(tint.rgb*alpha,alpha);}`;
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function init(){gl=canvas.getContext('webgl2',{antialias:false,alpha:false});if(!gl)throw Error('WebGL 2 is unavailable. Reference renders remain available; use a browser with hardware acceleration.');program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vertex));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.clearColor(.035,.045,.035,1);gl.disable(gl.DEPTH_TEST);indexBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,indexBuffer);gl.enableVertexAttribArray(0);gl.vertexAttribIPointer(0,1,gl.UNSIGNED_INT,4,0);gl.vertexAttribDivisor(0,1);}
function pose(view){position=[...view.position];forward=norm(view.forward);up=norm(view.up);fov=view.fov;center=add(position,mul(forward,.4));sortDirty=true;}
function fitOutside(){const data=manifest[current],bounds=data.bounds;if(!bounds){pose(data.views[0]);mode(false);return;}center=[...bounds.center];fov=55;const aspect=Math.max(.1,canvas.clientWidth/Math.max(1,canvas.clientHeight)),angle=Math.min(fov*Math.PI/360,Math.atan(Math.tan(fov*Math.PI/360)*aspect)),distance=bounds.radius/Math.sin(angle)*1.15;position=add(center,mul(norm([1,-1,.7]),distance));forward=norm(sub(center,position));up=norm(cross(norm(cross(forward,[0,0,1])),forward));mode(false);sortDirty=true;}
function stop(){playing=false;keys.clear();$('tour').textContent='Play tour';}
function mode(value){fly=value;$('mode').textContent=fly?'Enter orbit mode':'Enter fly mode';sortDirty=true;}
function controls(disabled){for(const id of ['reset','mode','tour','view'])$(id).disabled=disabled;}
async function load(name){const mine=++requestId;stop();controls(true);aborter?.abort();aborter=new AbortController();worker?.terminate();worker=null;busy=false;count=0;current=name;const data=manifest[name];history.replaceState(null,'','#'+name);for(const b of $('datasets').children)b.setAttribute('aria-pressed',String(b.dataset.name===name));$('title').textContent=name==='boot'?'Shoe':name[0].toUpperCase()+name.slice(1);$('type').textContent=name==='scene'?'CAPTURED SPACE / FREE FLIGHT':'OBJECT STUDY / ORBIT';$('detail').textContent=name==='scene'?'Explore the captured route. Free flight has no collision.':'Rotate, zoom and explore the captured viewpoints.';$('poster').src=data.poster||`assets/${name}.jpg`;$('poster').hidden=false;status.hidden=false;status.textContent=`Loading ${name} · ${((data.bytes+(data.sh_bytes||0))/1e6).toFixed(1)} MB`;
if(!gl){status.textContent='3D unavailable: enable WebGL 2 / hardware acceleration. Reference render shown.';return;}
try{
if(data.sh_degree!==3||data.sh_bytes!==data.count*128)throw Error('Full-color assets are missing. Run export_assets.py again.');
const [raw,shRaw]=await Promise.all([`${name}.bin`,`${name}.sh.bin`].map(async file=>{const res=await fetch(`${assetBase}/${file}`,{signal:aborter.signal});if(!res.ok)throw Error(`Model request failed (${res.status})`);return res.arrayBuffer();}));
if(mine!==requestId)return;
if(raw.byteLength!==data.count*64||shRaw.byteLength!==data.sh_bytes)throw Error('Model size mismatch. Run export_assets.py again.');
const floats=new Float32Array(raw),height=Math.ceil(data.count*4/2048);if(height>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('This model exceeds the device texture limit.');const padded=new Float32Array(2048*height*4);padded.set(floats);if(texture)gl.deleteTexture(texture);gl.activeTexture(gl.TEXTURE0);texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,2048,height,0,gl.RGBA,gl.FLOAT,padded);if(gl.getError()!==gl.NO_ERROR)throw Error('GPU texture allocation failed. Try a desktop browser.');
const shHeight=Math.ceil(data.count*16/2048);
if(shHeight>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('Full-color texture exceeds this device texture limit.');
const shPadded=new Uint16Array(2048*shHeight*4);shPadded.set(new Uint16Array(shRaw));
if(shTexture)gl.deleteTexture(shTexture);shTexture=gl.createTexture();gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,shTexture);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA16F,2048,shHeight,0,gl.RGBA,gl.HALF_FLOAT,shPadded);
if(gl.getError()!==gl.NO_ERROR)throw Error('Full-color GPU allocation failed. Try a desktop browser.');
gl.uniform1i(gl.getUniformLocation(program,'splats'),0);gl.uniform1i(gl.getUniformLocation(program,'harmonics'),1);

const centers=new Float32Array(data.count*3);for(let i=0;i<data.count;i++)centers.set(floats.subarray(i*16,i*16+3),i*3);worker=new Worker('sort-worker.js');worker.postMessage({centers:centers.buffer},[centers.buffer]);worker.onmessage=({data:sorted})=>{if(mine!==requestId)return;gl.bindBuffer(gl.ARRAY_BUFFER,indexBuffer);gl.bufferData(gl.ARRAY_BUFFER,sorted.order,gl.DYNAMIC_DRAW);count=data.count;busy=false;$('poster').hidden=true;status.hidden=true;controls(false);};worker.onerror=()=>{status.hidden=false;status.textContent='Depth sorting failed. Reload to retry.';};pose(data.views[0]);mode(name==='scene'&&!outside);if(outside)fitOutside();$('view').max=data.views.length-1;$('view').value=0;$('frame').textContent=`1 / ${data.views.length}`;$('stats').textContent=`${data.count.toLocaleString()} splats · ${((data.bytes+(data.sh_bytes||0))/1e6).toFixed(1)} MB`;sortDirty=true;}
catch(error){if(error.name==='AbortError')return;status.hidden=false;status.textContent=`${error.message}. Reference render shown.`;console.error(error);}}
function sort(now){if(!worker||busy||!sortDirty||now-lastSort<120)return;busy=true;sortDirty=false;lastSort=now;worker.postMessage({position,forward,sequence:++sequence});}
function draw(now){const dt=Math.min(.05,(now-lastTime)/1000||0);lastTime=now;if(position)sort(now);if(count){if(playing){tourTime+=dt;const views=manifest[current].views,at=tourTime/1.2;if(at>=views.length-1){stop();tourTime=0;}else{const i=Math.floor(at),a=views[i],b=views[i+1],t=at-i;position=add(mul(a.position,1-t),mul(b.position,t));forward=norm(add(mul(a.forward,1-t),mul(b.forward,t)));up=norm(add(mul(a.up,1-t),mul(b.up,t)));fov=a.fov*(1-t)+b.fov*t;$('view').value=i;$('frame').textContent=`${i+1} / ${views.length}`;sortDirty=true;}}
if(fly&&keys.size){const speed=dt*(keys.has('Shift')?1.2:.3),right=norm(cross(forward,up));let move=[0,0,0];for(const [key,dir] of [['w',forward],['s',mul(forward,-1)],['d',right],['a',mul(right,-1)],['e',[0,0,1]],['q',[0,0,-1]]])if(keys.has(key))move=add(move,dir);position=add(position,mul(move,speed));sortDirty=true;}
sort(now);const ratio=Math.min(devicePixelRatio,1.5),w=Math.max(1,Math.round(canvas.clientWidth*ratio)),h=Math.max(1,Math.round(canvas.clientHeight*ratio));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(program);const right=norm(cross(forward,up)),vertical=norm(cross(right,forward));for(const [key,value] of Object.entries({eye:position,right,up:vertical,forward}))gl.uniform3fv(gl.getUniformLocation(program,key),value);gl.uniform2f(gl.getUniformLocation(program,'viewport'),w,h);gl.uniform1f(gl.getUniformLocation(program,'focal'),h/(2*Math.tan(fov*Math.PI/360)));gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,shTexture);gl.bindBuffer(gl.ARRAY_BUFFER,indexBuffer);gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,count);}requestAnimationFrame(draw);}
let drag;
canvas.addEventListener('blur',stop);
canvas.addEventListener('pointerdown',e=>{if(!count)return;stop();canvas.focus();canvas.setPointerCapture(e.pointerId);drag=[e.clientX,e.clientY];});
canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=(e.clientX-drag[0])*.005,dy=(e.clientY-drag[1])*.005;drag=[e.clientX,e.clientY];const right=norm(cross(forward,up));if(e.shiftKey&&!fly){const delta=add(mul(right,-dx*.4),mul(up,dy*.4));position=add(position,delta);center=add(center,delta);}else if(fly){forward=norm(rotate(rotate(forward,[0,0,1],-dx),right,-dy));up=norm(cross(norm(cross(forward,[0,0,1])),forward));}else{let offset=sub(position,center);offset=rotate(rotate(offset,[0,0,1],-dx),right,-dy);position=add(center,offset);forward=norm(mul(offset,-1));up=norm(cross(norm(cross(forward,[0,0,1])),forward));}sortDirty=true;});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drag=null);
canvas.addEventListener('wheel',e=>{if(!count)return;e.preventDefault();stop();if(fly)position=add(position,mul(forward,-e.deltaY*.0005));else{const radius=manifest[current].bounds?.radius||.2,distance=Math.min(radius*100,Math.max(radius*.005,Math.hypot(...sub(position,center))*Math.exp(e.deltaY*.001)));position=sub(center,mul(forward,distance));}sortDirty=true;},{passive:false});
canvas.addEventListener('keydown',e=>{const key=e.key.length===1?e.key.toLowerCase():e.key;if(['w','a','s','d','q','e','Shift'].includes(key)){e.preventDefault();playing=false;$('tour').textContent='Play tour';keys.add(key);}});window.addEventListener('keyup',e=>keys.delete(e.key.length===1?e.key.toLowerCase():e.key));window.addEventListener('blur',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('keydown',e=>{if(e.key==='Escape')stop();});
$('reset').onclick=()=>{stop();if(outside)fitOutside();else pose(manifest[current].views[0]);$('view').value=0;$('frame').textContent=`1 / ${manifest[current].views.length}`;};$('mode').onclick=()=>{stop();center=add(position,mul(forward,.4));mode(!fly);};$('tour').onclick=()=>{playing=!playing;keys.clear();tourTime=Number($('view').value)*1.2;$('tour').textContent=playing?'Pause tour':'Play tour';};$('view').oninput=()=>{stop();const i=Number($('view').value);pose(manifest[current].views[i]);$('frame').textContent=`${i+1} / ${manifest[current].views.length}`;};$('help').onclick=()=>{stop();$('instructions').showModal();};$('close-help').onclick=()=>$('instructions').close();canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();count=0;stop();controls(true);status.hidden=false;status.textContent='Graphics context lost. Reload this page to restore the preview.';});
try{init();}catch(error){status.textContent=error.message;console.error(error);}
try{const res=await fetch(`${assetBase}/manifest.json`);if(!res.ok)throw Error('Assets not exported. Run export_assets.py first.');manifest=await res.json();for(const name of ['box','milk','boot','scene']){const button=document.createElement('button');button.dataset.name=name;button.innerHTML=`<img src="${manifest[name].poster||`assets/${name}.jpg`}" alt=""><span class="name">${name==='boot'?'Shoe':name[0].toUpperCase()+name.slice(1)}<small>${name==='scene'?'FLY':'ORBIT'}</small></span>`;button.onclick=()=>load(name);$('datasets').append(button);}await load(Object.hasOwn(manifest,location.hash.slice(1))?location.hash.slice(1):'box');requestAnimationFrame(draw);}catch(error){status.textContent=error.message;console.error(error);}

window.addEventListener('message',event=>{if(event.origin!==location.origin||event.data?.type!=='atlas-viewer')return;if(event.data.action==='fit'&&manifest&&current){stop();fitOutside();}if(event.data.action==='load'&&manifest&&Object.hasOwn(manifest,event.data.dataset)&&current!==event.data.dataset)load(event.data.dataset);});
