"""Check all correction variants in the served comparison page."""
import json
import argparse
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--scene',action='store_true');args=parser.parse_args()
port=9242 if args.scene else 9241
name='gnt-scene-15fps-splats.html' if args.scene else 'gnt-splat-correction.html'
profile='scene15_splat_qa' if args.scene else 'splat_correction_qa'
process=subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new','--disable-gpu',
    '--window-size=1700,1000',f'--remote-debugging-port={port}',
    '--user-data-dir='+str(Path.home()/'AppData/Local/Temp/sfm_lightweight_methods_v1'/profile),
    'http://localhost:8088/'+name],creationflags=subprocess.CREATE_NO_WINDOW,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list'))
        page=next(p for p in pages if name in p['url']);break
    except (OSError,StopIteration):
        if time.monotonic()>deadline:raise
        time.sleep(.5)
ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=180)
try:
    ws.send(json.dumps(dict(id=10,method='Page.enable')))
    while json.loads(ws.recv()).get('id')!=10:pass
    ws.send(json.dumps(dict(id=11,method='Page.reload',params=dict(ignoreCache=True))))
    while json.loads(ws.recv()).get('method')!='Page.loadEventFired':pass
    expression='''(async()=>{
      const check=(x,m)=>{if(!x)throw Error(m)},sleep=ms=>new Promise(r=>setTimeout(r,ms));
      const videos=[...document.querySelectorAll('video')];check(videos.map(v=>v.id).join(',')==='__PANELS__','Three panels');
      check(getComputedStyle(document.querySelector('.grid')).gridTemplateColumns.split(' ').length===3,'Three columns');
      const sources=new Set();
      for(const method of __METHODS__){
        const selector=document.getElementById('variant');selector.value=method;selector.dispatchEvent(new Event('change'));
        for(const v of videos){const until=Date.now()+15000;while(v.readyState<2&&Date.now()<until)await sleep(100);
          check(!v.error&&v.videoWidth===960&&v.videoHeight===540,'Decode');check(Math.abs(v.duration-__DURATION__)<.05,'Duration');
          const response=await fetch(v.src,{headers:{Range:'bytes=0-1023'}});check(response.status===206,'HTTP206');await response.arrayBuffer();sources.add(v.src);}
        check(videos[2].src.endsWith(method+'.mp4'),'Variant source');
        const seek=document.getElementById('seek');seek.value='2.433333333';seek.dispatchEvent(new Event('input'));await sleep(250);
        check(videos.every(v=>v.paused&&Math.abs(v.currentTime-2.433333333)<.05),'Seek');
        document.getElementById('play').click();await sleep(500);check(videos.every(v=>!v.paused),'Play');
        document.getElementById('pause').click();check(videos.every(v=>v.paused),'Pause');
      }
      const speed=document.getElementById('speed');speed.value='0.5';speed.dispatchEvent(new Event('change'));
      check(videos.every(v=>v.playbackRate===.5),'Slow playback');
      return {status:'passed',panels:3,videos:sources.size,checks:['labels/layout','decode/dimensions','duration','HTTP206','candidate variants','seek/play/pause','slow playback']};
    })()'''
    expression=expression.replace('__PANELS__','native,motion,splats' if args.scene else 'reference,baseline,candidate').replace('__METHODS__',json.dumps(['detail','fallback'] if args.scene else ['raw','colour','detail','fallback'])).replace('__DURATION__',str(2198/15 if args.scene else 6))
    ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
    while True:
        result=json.loads(ws.recv())
        if result.get('id')==1:break
    assert 'exceptionDetails' not in result['result'],result
    value=result['result']['result']['value']
    (ROOT/'outputs/gnt_finalization'/('scene15_splat_browser_qa.json' if args.scene else 'splat_correction_browser_qa.json')).write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)
finally:
    ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
