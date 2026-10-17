"""Headless browser check for the completed GNT-guided splat review."""
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1]
summary=json.loads((ROOT/'outputs/gnt_finalization/splat_fit.json').read_text())
expected_videos=1+len(summary['candidates'])+int(bool(summary['candidates'].get('mixed_10000',{}).get('full_route')))
expected_images=4*(1+len(summary['candidates']))
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new',
    '--disable-gpu','--window-size=1440,1000','--remote-debugging-port=9232',
    r'--user-data-dir=C:\Users\mopu01\AppData\Local\Temp\sfm_lightweight_methods_v1\splat_fit_browser_qa',
    'http://localhost:8088/gnt-splat-fit.html'],creationflags=subprocess.CREATE_NO_WINDOW,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen('http://127.0.0.1:9232/json/list'))
        page=next(p for p in pages if 'gnt-splat-fit.html' in p['url']);break
    except (OSError,StopIteration):
        if time.monotonic()>deadline:raise
        time.sleep(.5)
ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=120)
ws.send(json.dumps(dict(id=10,method='Page.enable')))
while json.loads(ws.recv()).get('id')!=10:pass
ws.send(json.dumps(dict(id=11,method='Page.reload',params=dict(ignoreCache=True))))
while json.loads(ws.recv()).get('method')!='Page.loadEventFired':pass
expression='''(async()=>{
const check=(x,m)=>{if(!x)throw Error(m)};
const media=[...document.querySelectorAll('video')];check(media.length===__VIDEOS__,'Video count');
for(const v of media){
 await new Promise((resolve,reject)=>{if(v.readyState>=1)resolve();else{v.onloadedmetadata=resolve;v.onerror=()=>reject(Error(v.src));setTimeout(()=>reject(Error('metadata timeout')),20000);}});
 check(v.videoWidth===960&&v.videoHeight===540,'Video dimensions');
 const full=v.src.endsWith('full_route.mp4');check(Math.abs(v.duration-(full?146.5:6))<.05,'Video duration');
 await new Promise(resolve=>{v.onseeked=resolve;v.currentTime=full?70:3;});
 check(v.readyState>=2,'Seek decoded');
}
const images=[...document.querySelectorAll('img')];
check(images.length===__IMAGES__,'Image count');
for(const image of images){image.loading='eager';await image.decode();check(image.naturalWidth===960&&image.naturalHeight===540,'Still dimensions');}
document.getElementById('play').click();await new Promise(resolve=>setTimeout(resolve,700));
check(media.filter(v=>!v.src.endsWith('full_route.mp4')).every(v=>!v.paused&&v.currentTime>0),'Play controls');
document.getElementById('pause').click();check(media.every(v=>v.paused),'Pause controls');
return {status:'passed',videos:media.length,images:images.length,checks:['metadata','dimensions','duration','seeking','image decoding','play','pause']};})()'''.replace('__VIDEOS__',str(expected_videos)).replace('__IMAGES__',str(expected_images))
ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
while True:
    result=json.loads(ws.recv())
    if result.get('id')==1:break
try:
    assert 'exceptionDetails' not in result['result'],result
    value=result['result']['result']['value'];output=ROOT/'outputs/gnt_finalization'
    (output/'splat_fit_browser_qa.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    ws.send(json.dumps(dict(id=2,method='Page.captureScreenshot',params=dict(format='jpeg',quality=85))))
    while True:
        screenshot=json.loads(ws.recv())
        if screenshot.get('id')==2:break
    (output/'splat_fit_browser_qa.jpg').write_bytes(base64.b64decode(screenshot['result']['data']))
    print(json.dumps(value),flush=True)
finally:
    ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
