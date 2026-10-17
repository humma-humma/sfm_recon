"""Check the traversal page in an isolated Chrome session on port 9231."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

parser = argparse.ArgumentParser()
parser.add_argument('--progress', action='store_true')
parser.add_argument('--wait', action='store_true', help='Wait for automatic export, then launch isolated Chrome.')
args = parser.parse_args()
if args.wait:
    assert not args.progress
    deadline = time.monotonic() + 10*60*60
    while True:
        with urllib.request.urlopen('http://localhost:8088/assets/gnt_traversal_preview/progress.json') as response:
            progress = json.load(response)
        if progress['published']:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Traversal did not publish within ten hours; inspect render progress.')
        time.sleep(30)
    subprocess.Popen([
        r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--headless=new', '--disable-gpu',
        '--window-size=1440,1000', '--remote-debugging-port=9231',
        r'--user-data-dir=C:\Users\mopu01\AppData\Local\Temp\sfm_lightweight_methods_v1\gnt_browser_qa',
        'http://localhost:8088/gnt-traversal.html'], creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic()+30
    while True:
        try:
            urllib.request.urlopen('http://127.0.0.1:9231/json/list').close()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(.5)
pages = json.load(urllib.request.urlopen('http://127.0.0.1:9231/json/list'))
page = next(p for p in pages if 'gnt-traversal.html' in p['url'])
ws = websocket.create_connection(page['webSocketDebuggerUrl'], suppress_origin=True, timeout=60)
ws.send(json.dumps(dict(id=10, method='Page.enable')))
while json.loads(ws.recv()).get('id') != 10:
    pass
ws.send(json.dumps(dict(id=11, method='Page.reload', params=dict(ignoreCache=True))))
while json.loads(ws.recv()).get('method') != 'Page.loadEventFired':
    pass
common = '''
const check=(x,m)=>{if(!x)throw Error(m)};
async function until(predicate){const deadline=Date.now()+30000;while(!predicate()){
 if(Date.now()>deadline)throw Error('Page readiness timeout');
 await new Promise(resolve=>setTimeout(resolve,100));}}
'''
if args.progress:
    expression = '''(async()=>{''' + common + '''
const image=document.getElementById('frame');
await until(()=>image.complete&&image.naturalWidth>0);
check(image.naturalWidth===960&&image.naturalHeight===540,'Preview dimensions');
check(document.getElementById('native').href===image.src,'Lossless preview link');
const progress=await(await fetch('assets/gnt_traversal_preview/progress.json')).json();
check(progress.completed>0&&progress.total===275&&!progress.published,'Progress state');
check(document.getElementById('status').textContent.includes('/275'),'Progress display');
return {status:'passed',completed:progress.completed,total:progress.total,
 checks:['live progress','native preview decoding','lossless frame link']};})()'''
else:
    expression = '''(async()=>{''' + common + '''
await until(()=>video.readyState>=2);
check(video.videoWidth===960&&video.videoHeight===540,'Video dimensions');
check(frames.length===275,'Full-route frame count');
check(Math.abs(video.duration-data.duration_seconds)<.1,'Video duration');
async function seek(time){video.currentTime=time;await until(()=>!video.seeking);update();}
await seek(0);document.getElementById('next').click();await until(()=>!video.seeking);
check(index()===1,'Next rendered view');
document.getElementById('previous').click();await until(()=>!video.seeking);
check(index()===0,'Previous rendered view');
const middle=Math.floor(frames.length/2);await seek(frames[middle].start+.1);
check(index()===middle,'Middle seek');
const native=document.getElementById('native');
check(native.href.endsWith(frames[middle].frame),'Lossless link follows seek');
const image=new Image();image.src=native.href;await image.decode();
check(image.naturalWidth===960&&image.naturalHeight===540,'Lossless dimensions');
const speed=document.getElementById('speed');speed.value='0.5';speed.dispatchEvent(new Event('change'));
check(video.playbackRate===.5,'Playback speed');
video.muted=true;await video.play();await until(()=>!video.paused);video.pause();
check(video.paused,'Pause');
await seek(frames.at(-1).start+.1);check(index()===274,'Last seek');
check((await fetch(document.querySelector('a[download]').href,{method:'HEAD'})).ok,'MP4 download');
await seek(frames[middle].start+.1);
return {status:'passed',views:frames.length,width:video.videoWidth,height:video.videoHeight,
 duration:video.duration,checks:['native decoding','full route','duration','next/previous',
 'middle/end seek','lossless frame link and decoding','speed','play/pause','MP4 download']};})()'''
ws.send(json.dumps(dict(id=1, method='Runtime.evaluate', params=dict(
    expression=expression, awaitPromise=True, returnByValue=True))))
while True:
    result = json.loads(ws.recv())
    if result.get('id') == 1:
        break
assert 'exceptionDetails' not in result['result'], result
value = result['result']['result']['value']
output = Path(__file__).resolve().parents[1]/'outputs/gnt_traversal'
output.mkdir(exist_ok=True)
name = 'progress_browser_qa' if args.progress else 'browser_qa'
(output/(name+'.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')
print(json.dumps(value))
ws.send(json.dumps(dict(id=90, method='Page.captureScreenshot', params=dict(format='jpeg', quality=90))))
while True:
    screenshot = json.loads(ws.recv())
    if screenshot.get('id') == 90:
        break
(output/(name+'.jpg')).write_bytes(base64.b64decode(screenshot['result']['data']))
ws.send(json.dumps(dict(id=99, method='Browser.close')))
ws.close()
