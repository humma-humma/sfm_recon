const data=JSON.parse(document.querySelector('#comparison-data').textContent);
const controls=Object.fromEntries(['dataset','view','left','right','wipe','zoom','status','comparison'].map(id=>[id,document.getElementById(id)]));
const context=controls.comparison.getContext('2d');
let images=null,center=[.5,.5],generation=0,drag=null;
const current=()=>data.cases[controls.dataset.value].find(c=>c.id===controls.view.value);
for(const name of Object.keys(data.cases))controls.dataset.add(new Option(name,name));
function methods(){
 for(const side of ['left','right']){
  const previous=controls[side].value;
  controls[side].replaceChildren();
  for(const name of Object.keys(current().urls))controls[side].add(new Option(data.labels[name],name));
  const preferred=side==='left'?(data.default_left||(current().urls["adapt1800"]?"adapt1800":(current().urls["control"]?"control":'gnt_base'))):(data.default_right||'baseline');
  controls[side].value=current().urls[previous]?previous:(current().urls[preferred]?preferred:Object.keys(current().urls)[0]);
 }
 load();
}
function views(){
 controls.view.replaceChildren();
 for(const c of data.cases[controls.dataset.value])controls.view.add(new Option(`${c.image} · ${c.kind==='motion'?c.id:`held out ${c.zoom}×${c.common_comparison?'':' · 2DGS only'}`}`,c.id));
 center=[.5,.5];controls.zoom.value='1';methods();
}
function draw(){
 if(!images)return;
 const c=current(),scale=Number(controls.zoom.value),w=c.width/scale,h=c.height/scale;
 center[0]=Math.max(.5/scale,Math.min(1-.5/scale,center[0]));
 center[1]=Math.max(.5/scale,Math.min(1-.5/scale,center[1]));
 const paint=img=>context.drawImage(img,center[0]*c.width-w/2,center[1]*c.height-h/2,w,h,0,0,960,540);
 paint(images[1]);
 const boundary=960*Number(controls.wipe.value)/100;
 context.save();context.beginPath();context.rect(0,0,boundary,540);context.clip();paint(images[0]);context.restore();
 if(boundary>0&&boundary<960){context.strokeStyle='#ffdf70';context.lineWidth=2;context.beginPath();context.moveTo(boundary,0);context.lineTo(boundary,540);context.stroke();}
 controls.status.textContent=`${data.labels[controls.left.value]} (left) / ${data.labels[controls.right.value]} (right) · ${c.width}×${c.height} · ${scale}× crop zoom`;
}
async function load(){
 const token=++generation,c=current();images=null;context.clearRect(0,0,960,540);controls.status.textContent='Loading matching images…';
 try{
  const loaded=await Promise.all(['left','right'].map(async side=>{
   const image=new Image();image.src=c.urls[controls[side].value];await image.decode();
   if(image.naturalWidth!==c.width||image.naturalHeight!==c.height)throw new Error('Unexpected image dimensions');return image;
  }));
  if(token!==generation)return;images=loaded;draw();
 }catch(error){if(token===generation)controls.status.textContent='Unable to load comparison: '+error.message;}
}
controls.dataset.addEventListener('change',views);controls.view.addEventListener('change',methods);
for(const side of ['left','right'])controls[side].addEventListener('change',load);
controls.wipe.addEventListener('input',draw);controls.zoom.addEventListener('input',draw);
document.getElementById('toggle').addEventListener('click',()=>{controls.wipe.value=Number(controls.wipe.value)>=50?'0':'100';draw();});
document.getElementById('reset').addEventListener('click',()=>{center=[.5,.5];controls.zoom.value='1';draw();});
controls.comparison.addEventListener('pointerdown',e=>{drag={id:e.pointerId,x:e.clientX,y:e.clientY};controls.comparison.setPointerCapture(e.pointerId);});
controls.comparison.addEventListener('pointermove',e=>{
 if(!drag||e.pointerId!==drag.id)return;
 const rect=controls.comparison.getBoundingClientRect(),scale=Number(controls.zoom.value);
 center[0]-=(e.clientX-drag.x)/(rect.width*scale);center[1]-=(e.clientY-drag.y)/(rect.height*scale);
 drag.x=e.clientX;drag.y=e.clientY;draw();
});
for(const event of ['pointerup','pointercancel','lostpointercapture'])controls.comparison.addEventListener(event,()=>{drag=null;});
views();
