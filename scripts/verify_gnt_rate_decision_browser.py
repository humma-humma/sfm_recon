"""Verify decision-test video delivery, playback controls and held/motion modes."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket
import argparse

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
group.add_argument('--fifteen',action='store_true');group.add_argument('--splats',action='store_true');args=parser.parse_args()
name='gnt-splat-interleaved.html' if args.splats else 'gnt-rate-decision-15.html' if args.fifteen else 'gnt-rate-decision.html'
port=9239 if args.splats else 9238 if args.fifteen else 9237
profile='splat_interleaved_qa' if args.splats else 'rate_decision_15_qa' if args.fifteen else 'rate_decision_qa'
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new','--disable-gpu',
    '--window-size=1600,1000',f'--remote-debugging-port={port}',
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
    const videos=[...document.querySelectorAll('video')];check(videos.map(v=>v.id).join(',')==='__PANELS__','Three labelled panels');
    check(getComputedStyle(document.querySelector('.grid')).gridTemplateColumns.split(' ').length===3,'Aligned layout');
    const unique=new Set();
    for(const mode of __MODES__){
      if(__SPLATS__){for(const key of ['left','right']){const select=document.getElementById(key+'_rate');select.value=mode;select.dispatchEvent(new Event('change'));}}
      else {const select=document.getElementById('mode');select.value=mode;select.dispatchEvent(new Event('change'));}
      for(const v of videos){const until=Date.now()+15000;while(v.readyState<2&&Date.now()<until)await sleep(100);
        check(!v.error&&v.videoWidth===960&&v.videoHeight===540,'Video decode/dimensions');check(Math.abs(v.duration-6)<.05,'Duration');
        check(v.id==='reference'||v.src.endsWith(__SPLATS__?'fps'+mode+'_splat.mp4':v.id+'_'+mode+'.mp4'),'Mode source');
        const response=await fetch(v.src,{headers:{Range:'bytes=0-1023'}});check(response.status===206,'Byte-range access');await response.arrayBuffer();unique.add(v.src);
      }
      const slider=document.getElementById('seek');slider.value='2.433333333';slider.dispatchEvent(new Event('input'));await sleep(300);
      check(videos.every(v=>Math.abs(v.currentTime-2.433333333)<.05&&v.paused),'Shared seek');
      document.getElementById('play').click();await sleep(500);check(videos.every(v=>!v.paused),'Play all');
      document.getElementById('pause').click();await sleep(100);check(videos.every(v=>v.paused),'Pause all');
    }
    const speed=document.getElementById('speed');speed.value='0.5';speed.dispatchEvent(new Event('change'));check(videos.every(v=>v.playbackRate===.5),'Slow playback');
    return {status:'passed',panels:3,videos:unique.size,checks:['aligned labels/panels','dimensions/decode','duration','HTTP206','candidate switches','seek/play/pause','slow playback']};})()'''
    expression=expression.replace('__PANELS__','reference,left,right' if args.splats else 'reference,fps10,fps15' if args.fifteen else 'reference,fps5,fps10')
    expression=expression.replace('__SPLATS__','true' if args.splats else 'false').replace('__MODES__',json.dumps(['5','10','15'] if args.splats else ['held','motion']))
    ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
    while True:
        result=json.loads(ws.recv())
        if result.get('id')==1:break
    assert 'exceptionDetails' not in result['result'],result
    value=result['result']['result']['value']
    output='splat_interleaved_browser_qa.json' if args.splats else 'rate_decision_15_browser_qa.json' if args.fifteen else 'rate_decision_browser_qa.json'
    (ROOT/'outputs/gnt_finalization'/output).write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)
finally:
    ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
