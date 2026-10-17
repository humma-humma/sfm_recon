"""Check live progress or both completed full-scene GNT exports."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');args=parser.parse_args()
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new','--disable-gpu',
    '--window-size=1600,1000','--remote-debugging-port=9240',
    '--user-data-dir='+str(Path.home()/'AppData/Local/Temp/sfm_lightweight_methods_v1/scene15_qa'),
    'http://localhost:8088/gnt-scene-15fps.html'],creationflags=subprocess.CREATE_NO_WINDOW,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen('http://127.0.0.1:9240/json/list'))
        page=next(p for p in pages if 'gnt-scene-15fps.html' in p['url']);break
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
    const until=Date.now()+15000;while(document.getElementById('progress').textContent==='Checking progress…'&&Date.now()<until)await sleep(100);
    const response=await fetch('assets/gnt_finalization/scene15/progress.json',{cache:'no-store'});check(response.ok,'Progress delivery');const progress=await response.json();
    check(progress.total===2198&&progress.completed<=progress.total,'Progress counts');
    if(__LIVE__){check(document.getElementById('progress').textContent.includes(progress.completed+'/2198'),'Live status');
      const preview=document.getElementById('preview');if(!document.getElementById('preview_box').hidden){await preview.decode();check(preview.naturalWidth===960&&preview.naturalHeight===540,'Preview');}
      return {status:'passed',live:true,completed:progress.completed,reused:progress.reused,checks:['progress delivery','counts','live status','preview if available']};}
    check(progress.status==='complete','Render/export complete');const videos=[...document.querySelectorAll('video')];check(videos.map(v=>v.id).join(',')==='native,motion','Two exports');
    for(const v of videos){const deadline=Date.now()+15000;while(v.readyState<2&&Date.now()<deadline)await sleep(100);
      check(!v.error&&v.videoWidth===960&&v.videoHeight===540,'Dimensions/decode');check(Math.abs(v.duration-2198/15)<.05,'Duration');
      const r=await fetch(v.src,{headers:{Range:'bytes=0-1023'}});check(r.status===206,'Range access');await r.arrayBuffer();}
    const seek=document.getElementById('seek');seek.value='73.266666667';seek.dispatchEvent(new Event('input'));await sleep(300);check(videos.every(v=>Math.abs(v.currentTime-73.266666667)<.05&&v.paused),'Shared seek');
    document.getElementById('play').click();await sleep(500);check(videos.every(v=>!v.paused),'Play both');document.getElementById('pause').click();await sleep(100);check(videos.every(v=>v.paused),'Pause both');
    const speed=document.getElementById('speed');speed.value='0.5';speed.dispatchEvent(new Event('change'));check(videos.every(v=>v.playbackRate===.5),'Slow playback');
    return {status:'passed',live:false,videos:2,checks:['progress','dimensions/decode','duration','HTTP206','seek/play/pause','slow playback']};})()'''.replace('__LIVE__','true' if args.live else 'false')
    ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
    while True:
        result=json.loads(ws.recv())
        if result.get('id')==1:break
    assert 'exceptionDetails' not in result['result'],result
    value=result['result']['result']['value']
    file='scene15_live_browser_qa.json' if args.live else 'scene15_browser_qa.json'
    (ROOT/'outputs/gnt_finalization'/file).write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)
finally:
    ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
