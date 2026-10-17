"""Verify the five-video portfolio and capture its rendered desktop layout."""
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1]
port=9243
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new','--disable-gpu',
    '--window-size=1700,1600',f'--remote-debugging-port={port}',
    '--user-data-dir='+str(Path.home()/'AppData/Local/Temp/sfm_lightweight_methods_v1/portfolio_qa'),
    'http://localhost:8088/portfolio.html'],creationflags=subprocess.CREATE_NO_WINDOW,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list'))
        page=next(p for p in pages if 'portfolio.html' in p['url']);break
    except (OSError,StopIteration):
        if time.monotonic()>deadline:raise
        time.sleep(.5)
ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=180)
def call(i,method,params=None):
    ws.send(json.dumps(dict(id=i,method=method,params=params or {})))
    while True:
        value=json.loads(ws.recv())
        if value.get('id')==i:return value['result']
try:
    call(1,'Page.enable');call(2,'Page.reload',dict(ignoreCache=True))
    expression='''(async()=>{
      const check=(x,m)=>{if(!x)throw Error(m)},sleep=ms=>new Promise(r=>setTimeout(r,ms));
      for(let i=0;i<100&&document.querySelectorAll('video').length!==5;i++)await sleep(100);
      const videos=[...document.querySelectorAll('video')];check(videos.map(v=>v.id).join(',')==='box,milk,shoe,scene-motion,scene-splats','Five labelled videos');
      check(getComputedStyle(document.querySelector('.objects')).gridTemplateColumns.split(' ').length===3,'Object row');
      check(getComputedStyle(document.querySelector('.scenes')).gridTemplateColumns.split(' ').length===2,'Scene row');
      for(const [i,v] of videos.entries()){
        const until=Date.now()+15000;while(v.readyState<2&&Date.now()<until)await sleep(100);
        check(!v.error&&v.videoWidth===960&&v.videoHeight===540,'Video dimensions/decode');check(Math.abs(v.duration-(i<3?32:2198/15))<.05,'Full-route duration');
        const range=await fetch(v.src,{headers:{Range:'bytes=0-1023'}});check(range.status===206,'HTTP206');await range.arrayBuffer();
        const poster=await fetch(v.poster);check(poster.ok,'Poster');
      }
      document.getElementById('play-all').click();await sleep(700);check(videos.every(v=>!v.paused&&!v.error),'Play all');
      document.getElementById('pause-all').click();await sleep(150);check(videos.every(v=>v.paused),'Pause all');
      const speed=document.getElementById('speed');speed.value='0.5';speed.dispatchEvent(new Event('change'));check(videos.every(v=>v.playbackRate===.5),'Speed');
      const scenes=videos.slice(3),seek=document.getElementById('scene-seek');seek.value='12.5';seek.dispatchEvent(new Event('input'));await sleep(350);check(scenes.every(v=>Math.abs(v.currentTime-12.5)<.05&&v.paused),'Scene shared seek');
      scenes[1].currentTime=46;await sleep(400);check(Math.abs(scenes[0].currentTime-46)<.08,'Individual scene seek sync');
      document.getElementById('play-scenes').click();await sleep(600);check(scenes.every(v=>!v.paused),'Scene pair play');
      document.getElementById('pause-scenes').click();await sleep(150);check(scenes.every(v=>v.paused),'Scene pair pause');
      speed.value='1';speed.dispatchEvent(new Event('change'));videos.forEach(v=>v.currentTime=0);await sleep(300);
      return {status:'passed',videos:5,checks:['five labels','desktop layout','full durations','decode/dimensions','HTTP206','posters','play/pause all','speed','scene shared and individual seeking','scene paired playback']};
    })()'''
    result=call(3,'Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True));assert 'exceptionDetails' not in result,result
    report=result['result']['value']
    image=call(4,'Page.captureScreenshot',dict(format='png',captureBeyondViewport=True))
    (ROOT/'outputs/gnt_finalization/portfolio_desktop.png').write_bytes(base64.b64decode(image['data']))
    call(5,'Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
    result=call(6,'Runtime.evaluate',dict(expression="new Promise(r=>setTimeout(()=>r({overflow:document.documentElement.scrollWidth>innerWidth,objects:getComputedStyle(document.querySelector('.objects')).gridTemplateColumns.split(' ').length,scenes:getComputedStyle(document.querySelector('.scenes')).gridTemplateColumns.split(' ').length}),300))",awaitPromise=True,returnByValue=True))
    mobile=result['result']['value'];assert not mobile['overflow'] and mobile['objects']==mobile['scenes']==1,mobile
    report['mobile']=mobile
    (ROOT/'outputs/gnt_finalization/portfolio_browser_qa.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    call(99,'Browser.close');ws.close()
