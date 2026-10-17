"""Exercise final route navigation and absence of the removed orbit mode."""
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1];port=9244
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new',
    '--use-angle=d3d11','--enable-webgl','--ignore-gpu-blocklist','--window-size=1500,1200',f'--remote-debugging-port={port}',
    '--user-data-dir='+str(Path.home()/'AppData/Local/Temp/sfm_lightweight_methods_v1/final_demo_qa'),
    'http://localhost:8088/?mode=outside#scene'],creationflags=subprocess.CREATE_NO_WINDOW,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list'));page=next(p for p in pages if p['url'].startswith('http://localhost:8088/'));break
    except (OSError,StopIteration):
        if time.monotonic()>deadline:raise
        time.sleep(.5)
ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=240)
def call(i,method,params=None):
    ws.send(json.dumps(dict(id=i,method=method,params=params or {})))
    while True:
        value=json.loads(ws.recv())
        if value.get('id')==i:
            assert 'error' not in value,value
            return value['result']
def evaluate(i,expression):
    result=call(i,'Runtime.evaluate',dict(expression=expression,awaitPromise=True,returnByValue=True));assert 'exceptionDetails' not in result,result
    return result['result'].get('value')
try:
    call(1,'Page.enable');call(2,'Page.reload',dict(ignoreCache=True))
    report=evaluate(3,'''(async()=>{
      const check=(x,m)=>{if(!x)throw Error(m)},sleep=ms=>new Promise(r=>setTimeout(r,ms));
      const ready=async()=>{const until=Date.now()+20000;while((!document.getElementById('route')||document.getElementById('route').readyState<2)&&Date.now()<until)await sleep(100);};await ready();
      const video=document.getElementById('route');check(video.src.endsWith('scene15_motion.mp4'),'Default motion scene');check(!document.querySelector('iframe,#outside-mode,#fit'),'Orbit UI removed');check(!new URL(location.href).searchParams.has('mode'),'Old outside-mode bookmark normalizes to route');check(!performance.getEntriesByType('resource').some(e=>/interactive_models|[.]bin|orbit[.]html/.test(e.name)),'No live model loading');
      const routes=[];
      for(const name of ['box','milk','boot','scene']){
        document.querySelector('[data-dataset="'+name+'"]').click();await ready();check(video.videoWidth===960&&video.videoHeight===540,'Video decode');check(Math.abs(video.duration-(name==='scene'?2198/15:32))<.05,'Full duration');routes.push(name);
      }
      const seek=document.getElementById('seek');seek.value='43.2';seek.dispatchEvent(new Event('input'));await sleep(200);
      const renderer=document.getElementById('renderer');renderer.value='fallback';renderer.dispatchEvent(new Event('change'));await ready();await sleep(200);check(video.src.endsWith('/fallback.mp4')&&Math.abs(video.currentTime-43.2)<.05,'Switch renderer preserves time');
      renderer.value='motion';renderer.dispatchEvent(new Event('change'));await ready();await sleep(200);check(video.src.endsWith('scene15_motion.mp4'),'Return to preferred motion');
      document.getElementById('play').click();await sleep(400);check(!video.paused,'Play');document.getElementById('play').click();check(video.paused,'Pause');
      const viewport=document.getElementById('viewport');viewport.focus();const start=video.currentTime;viewport.dispatchEvent(new KeyboardEvent('keydown',{key:'w',bubbles:true}));await sleep(650);window.dispatchEvent(new KeyboardEvent('keyup',{key:'w'}));check(video.currentTime>start+.35,'Forward navigation');
      const forward=video.currentTime;viewport.dispatchEvent(new KeyboardEvent('keydown',{key:'s',bubbles:true}));await sleep(400);window.dispatchEvent(new KeyboardEvent('keyup',{key:'s'}));check(video.currentTime<forward-.2,'Backward navigation');
      document.getElementById('reset').click();check(video.currentTime<.05,'Reset route');
      return {status:'passed',routes,checks:['orbit removed','old outside bookmark uses route','no live model loading','motion default','four full routes','renderer switch preserves time','play/pause','keyboard forward/back','route reset']};
    })()''')
    evaluate(111,"(async()=>{const v=document.getElementById('route'),until=Date.now()+15000;while((v.readyState<3||v.seeking)&&Date.now()<until)await new Promise(r=>setTimeout(r,100));if(v.readyState<3)throw Error('Route did not finish loading');return true;})()")
    image=call(112,'Page.captureScreenshot',dict(format='png',captureBeyondViewport=True))
    (ROOT/'outputs/gnt_finalization/final_demo_desktop.png').write_bytes(base64.b64decode(image['data']))
    call(113,'Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
    mobile=evaluate(114,"new Promise(r=>setTimeout(()=>r({overflow:document.documentElement.scrollWidth>innerWidth,columns:getComputedStyle(document.getElementById('datasets')).gridTemplateColumns.split(' ').length}),300))")
    assert not mobile['overflow'] and mobile['columns']==2,mobile;report['mobile']=mobile
    (ROOT/'outputs/gnt_finalization/final_demo_browser_qa.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
finally:
    call(999,'Browser.close');ws.close()
