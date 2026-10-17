const $=id=>document.getElementById(id),video=$('route'),viewport=$('viewport');
const names={box:'Box',milk:'Milk',boot:'Shoe',scene:'Scene'};
let dataset=Object.hasOwn(names,location.hash.slice(1))?location.hash.slice(1):'scene',choice='motion',request=0,drag=null,lastSeek=0;
const keys=new Set();
function url(){return dataset==='scene'?(choice==='motion'?'assets/gnt_finalization/scene15/scene15_motion.mp4':'assets/gnt_finalization/scene15_splat/fallback.mp4'):`assets/gnt_finalization/balanced_rollout/${dataset}/${dataset}_balanced90.mp4`;}
function updateTime(){const duration=Number.isFinite(video.duration)?video.duration:dataset==='scene'?2198/15:32;$('seek').max=Math.max(0,duration-1/30);$('seek').value=video.currentTime;$('time').textContent=video.currentTime.toFixed(2)+' / '+duration.toFixed(2)+' s';}
function stop(){keys.clear();video.pause();drag=null;}
function seek(time){video.currentTime=Math.max(0,Math.min(Number.isFinite(video.duration)?video.duration-1/30:0,time));updateTime();}
async function togglePlay(){if(video.paused){try{await video.play();}catch(e){$('loading').hidden=false;$('loading').textContent='Playback could not start. Try Play route again.';}}else video.pause();}
function update(preserve=false){const oldTime=preserve?video.currentTime:0,wasPlaying=preserve&&!video.paused;stop();const mine=++request;
  for(const b of $('datasets').children)b.setAttribute('aria-pressed',String(b.dataset.dataset===dataset));
  $('renderer-label').hidden=dataset!=='scene';$('title').textContent=names[dataset];
  const locationUrl=new URL(location.href);locationUrl.hash=dataset;locationUrl.searchParams.delete('mode');history.replaceState(null,'',locationUrl);
    $('loading').hidden=false;$('loading').textContent='Loading '+names[dataset]+'…';video.poster='assets/portfolio/'+(dataset==='scene'?'scene_'+(choice==='motion'?'motion':'splats'):dataset==='boot'?'shoe':dataset)+'.jpg';
    video.onloadedmetadata=()=>{if(mine!==request)return;seek(oldTime);video.playbackRate=+$('speed').value;updateTime();};
    video.onloadeddata=()=>{if(mine!==request)return;$('loading').hidden=true;if(wasPlaying)video.play().catch(()=>{});};video.src=url();video.load();
    $('badge').textContent=dataset==='scene'?(choice==='motion'?'GNT + motion compensation · 30fps':'Corrected splats + reliability fallback · 30fps'):'Balanced Gaussian splats · 30fps';
    $('description').textContent=dataset==='scene'?(choice==='motion'?'The full building route using 15 independent GNT views per second, motion compensated to 30fps.':'The same route with corrected splat frames between GNT anchors; unreliable regions use motion interpolation.'):'The full 32-second captured camera route, rendered with the final balanced Gaussian model.';
    $('controls-hint').textContent='Click the view, then W/S or ↑/↓ to travel · Shift moves faster · Space plays/pauses · Drag to scrub';$('scope').textContent='Route navigation follows pre-rendered camera positions. Scene defaults to motion compensation; the splat alternative remains available in the render selector.';
}
for(const button of $('datasets').children)button.onclick=()=>{dataset=button.dataset.dataset;update();};
$('renderer').onchange=e=>{choice=e.target.value;update(true);};
$('play').onclick=togglePlay;$('reset').onclick=()=>{stop();seek(0);};$('seek').oninput=e=>{stop();seek(+e.target.value);};$('speed').onchange=e=>video.playbackRate=+e.target.value;
$('fullscreen').onclick=()=>video.requestFullscreen?.();video.addEventListener('timeupdate',updateTime);video.addEventListener('play',()=>$('play').textContent='Pause route');video.addEventListener('pause',()=>$('play').textContent='Play route');video.addEventListener('error',()=>{if(video.getAttribute('src')){$('loading').hidden=false;$('loading').textContent='Video unavailable. Check that the local artifact drive is connected.';}});
viewport.addEventListener('keydown',e=>{if(e.target.closest('input,button,select'))return;const key=e.key.toLowerCase();if(['w','s','arrowup','arrowdown','shift'].includes(key)){e.preventDefault();video.pause();if(!keys.has(key))lastSeek=performance.now();keys.add(key);}if(key===' '){e.preventDefault();togglePlay();}if(key==='arrowleft'||key==='arrowright'){e.preventDefault();video.pause();seek(video.currentTime+(key==='arrowright'?1:-1)/30);}if(key==='escape')stop();});
window.addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));window.addEventListener('blur',stop);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});viewport.addEventListener('focusout',e=>{if(!viewport.contains(e.relatedTarget))keys.clear();});
viewport.addEventListener('pointerdown',e=>{if(e.button!==0)return;video.pause();viewport.focus();viewport.setPointerCapture(e.pointerId);drag={x:e.clientX,time:video.currentTime};});
viewport.addEventListener('pointermove',e=>{if(drag&&Number.isFinite(video.duration))seek(drag.time+(e.clientX-drag.x)/Math.max(1,video.clientWidth)*video.duration);});for(const event of ['pointerup','pointercancel','lostpointercapture'])viewport.addEventListener(event,()=>drag=null);
function tick(now){if(keys.size&&now-lastSeek>=80){const direction=Number(keys.has('w')||keys.has('arrowup'))-Number(keys.has('s')||keys.has('arrowdown'));if(direction)seek(video.currentTime+direction*(now-lastSeek)/1000*(keys.has('shift')?3:1));lastSeek=now;}requestAnimationFrame(tick);}update();requestAnimationFrame(tick);
