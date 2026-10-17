"""Verify every published object comparison in an isolated headless browser."""
import base64
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1]
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new',
    '--disable-gpu','--window-size=1440,1000','--remote-debugging-port=9231',
    r'--user-data-dir=C:\Users\mopu01\AppData\Local\Temp\sfm_lightweight_methods_v1\gnt_browser_qa',
    'http://localhost:8088/gnt-objects.html'],creationflags=subprocess.CREATE_NO_WINDOW,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen('http://127.0.0.1:9231/json/list'))
        page=next(p for p in pages if 'gnt-objects.html' in p['url']);break
    except (OSError,StopIteration):
        if time.monotonic()>deadline:raise
        time.sleep(.5)
ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=120)
ws.send(json.dumps(dict(id=10,method='Page.enable')))
while json.loads(ws.recv()).get('id')!=10:pass
ws.send(json.dumps(dict(id=11,method='Page.reload',params=dict(ignoreCache=True))))
while json.loads(ws.recv()).get('method')!='Page.loadEventFired':pass
expression='''(async()=>{
const check=(x,m)=>{if(!x)throw Error(m)};let choices=0,viewCount=0;
check(Object.keys(data.cases).length===3,'Missing dataset');
for(const name of Object.keys(data.cases)){
 controls.dataset.value=name;views();
 for(const c of data.cases[name]){
  controls.view.value=c.id;methods();viewCount++;
  for(const label of Object.keys(c.urls)){
   controls.left.value=label;await load();
   check(images&&images.every(image=>image.naturalWidth===c.width&&image.naturalHeight===c.height),'Decoded dimensions');
   check(controls.left.value===label,'Method selection');choices++;
  }
  check(c.kind!=='motion'||!c.urls.gt,'Virtual reference photo');
 }
}
controls.dataset.value=Object.keys(data.cases)[0];views();
controls.left.value='balanced6000';controls.right.value='pretrained';await load();
check(controls.left.value==='balanced6000'&&controls.right.value==='pretrained','Default methods');
controls.zoom.value='2';draw();check(center.every(x=>x>=.25&&x<=.75),'Crop bounds');
controls.wipe.value='25';draw();document.getElementById('toggle').click();check(controls.wipe.value==='100','Toggle');
document.getElementById('reset').click();check(controls.zoom.value==='1','Reset crop');
controls.wipe.value='50';draw();
return {status:'passed',datasets:Object.keys(data.cases),views:viewCount,choices,
 checks:['all images decoded','dimensions','all method choices','default methods','crop bounds','toggle','reset','virtual reference exclusion']};})()'''
ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
while True:
    result=json.loads(ws.recv())
    if result.get('id')==1:break
assert 'exceptionDetails' not in result['result'],result
value=result['result']['result']['value'];output=ROOT/'outputs/gnt_finalization'
(output/'object_browser_qa.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
print(json.dumps(value),flush=True)
ws.send(json.dumps(dict(id=90,method='Page.captureScreenshot',params=dict(format='jpeg',quality=90))))
while True:
    result=json.loads(ws.recv())
    if result.get('id')==90:break
(output/'object_browser_qa.jpg').write_bytes(base64.b64decode(result['result']['data']))
ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
