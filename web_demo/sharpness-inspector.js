// Shared source-pixel crop: both images always use the same transform.
const inspection=JSON.parse(document.querySelector('#inspect-data').textContent);
const dataset=document.querySelector('#inspect-dataset');
const view=document.querySelector('#inspect-case');
const method=document.querySelector('#inspect-method');
const wipe=document.querySelector('#inspect-wipe');
const zoom=document.querySelector('#inspect-zoom');
const status=document.querySelector('#inspect-status');
const canvas=document.querySelector('#inspect-canvas');
const context=canvas.getContext('2d');
let images=null,center=[480,270],generation=0,drag=null;
for(const name of Object.keys(inspection.cases))dataset.add(new Option(name,name));
for(const [key,label] of Object.entries(inspection.labels)){
 if(key!=='baseline')method.add(new Option(label,key));
}
method.value='surface_photo';
function populateViews(){
 view.replaceChildren();
 for(const item of inspection.cases[dataset.value])view.add(new Option(`${item.image} · ${item.zoom}×`,item.id));
 center=[480,270];zoom.value='1';load();
}
function draw(){
 if(!images)return;
 const width=960/Number(zoom.value),height=540/Number(zoom.value);
 center[0]=Math.max(width/2,Math.min(960-width/2,center[0]));
 center[1]=Math.max(height/2,Math.min(540-height/2,center[1]));
 const paint=image=>context.drawImage(image,center[0]-width/2,center[1]-height/2,width,height,0,0,960,540);
 paint(images.original);
 const boundary=960*Number(wipe.value)/100;
 context.save();context.beginPath();context.rect(0,0,boundary,540);context.clip();paint(images.candidate);context.restore();
 if(boundary>0&&boundary<960){context.strokeStyle='#ffdf70';context.lineWidth=2;context.beginPath();context.moveTo(boundary,0);context.lineTo(boundary,540);context.stroke();}
 status.textContent=`${inspection.labels[method.value]} (left) / Original splats (right) · ${zoom.value}× crop zoom · ${wipe.value}% candidate`;
}
async function load(){
 const token=++generation;
 images=null;context.clearRect(0,0,960,540);status.textContent='Loading matching images…';
 const base=`assets/sharpness/${dataset.value}/${view.value}_`;
 try{
  const [original,candidate]=await Promise.all(['baseline',method.value].map(async name=>{
   const image=new Image();image.src=base+name+'.jpg';await image.decode();
   if(image.naturalWidth!==960||image.naturalHeight!==540)throw new Error('Unexpected image dimensions');return image;
  }));
  if(token!==generation)return;images={original,candidate};draw();
 }catch(error){if(token===generation)status.textContent='Unable to load comparison: '+error.message;}
}
dataset.addEventListener('change',populateViews);
view.addEventListener('change',load);method.addEventListener('change',load);
wipe.addEventListener('input',draw);zoom.addEventListener('input',draw);
document.querySelector('#inspect-toggle').addEventListener('click',()=>{wipe.value=Number(wipe.value)>=50?'0':'100';draw();});
document.querySelector('#inspect-reset').addEventListener('click',()=>{center=[480,270];zoom.value='1';draw();});
canvas.addEventListener('pointerdown',event=>{drag={id:event.pointerId,x:event.clientX,y:event.clientY};canvas.setPointerCapture(event.pointerId);canvas.style.cursor='grabbing';});
canvas.addEventListener('pointermove',event=>{
 if(!drag||event.pointerId!==drag.id)return;
 const rect=canvas.getBoundingClientRect(),scale=Number(zoom.value);
 center[0]-=(event.clientX-drag.x)*960/(rect.width*scale);
 center[1]-=(event.clientY-drag.y)*540/(rect.height*scale);
 drag.x=event.clientX;drag.y=event.clientY;draw();
});
for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{drag=null;canvas.style.cursor='grab';});
populateViews();
