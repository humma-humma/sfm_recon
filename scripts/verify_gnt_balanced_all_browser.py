"""Validate all four completed videos, comparison clips and synchronized controls."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1]
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new','--disable-gpu',
    '--window-size=1600,1000','--remote-debugging-port=9236',
    '--user-data-dir='+str(Path.home()/'AppData/Local/Temp/sfm_lightweight_methods_v1/balanced_all_qa'),
    'http://localhost:8088/gnt-balanced-all.html'],creationflags=subprocess.CREATE_NO_WINDOW,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen('http://127.0.0.1:9236/json/list'))
        page=next(p for p in pages if 'gnt-balanced-all.html' in p['url']);break
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
    const videos=[...document.querySelectorAll('video')];check(videos.length===16,'All four exports');
    for(const v of videos){v.muted=true;const until=Date.now()+15000;while(v.readyState<1&&Date.now()<until)await sleep(100);
      check(!v.error&&v.videoWidth===960&&v.videoHeight===540,'Dimensions/decode');
      check(v.classList.contains('comparison')?Math.abs(v.duration-6)<.05:[32,146.5].some(d=>Math.abs(v.duration-d)<.05),'Duration');
      const response=await fetch(v.src,{headers:{Range:'bytes=0-1023'}});check(response.status===206,'Range support');await response.arrayBuffer();
    }
    for(const group of document.querySelectorAll('.triple')){const clips=[...group.querySelectorAll('video')];
      check(clips.length===3,'Three comparison panels');clips[0].currentTime=2;await sleep(400);
      check(clips.every(v=>Math.abs(v.currentTime-2)<.15),'Synchronized seek');
      await clips[0].play();await sleep(300);check(clips.every(v=>!v.paused),'Synchronized play');
      clips[0].pause();await sleep(100);check(clips.every(v=>v.paused),'Synchronized pause');
    }
    for(const v of document.querySelectorAll('video.full')){v.currentTime=v.duration/2;await sleep(200);await v.play();await sleep(100);v.pause();check(!v.error,'Full video playback');}
    return {status:'passed',videos:videos.length,checks:['dimensions/decode','duration','HTTP206','three panels','synchronized seek/play/pause','full video playback']};})()'''
    ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
    while True:
        result=json.loads(ws.recv())
        if result.get('id')==1:break
    assert 'exceptionDetails' not in result['result'],result
    value=result['result']['result']['value']
    (ROOT/'outputs/gnt_finalization/balanced_all_browser_qa.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)
finally:
    ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
