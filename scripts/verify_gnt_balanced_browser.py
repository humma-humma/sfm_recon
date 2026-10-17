"""Exercise the balanced comparison in an isolated Chrome session on port 9231."""
import argparse
import base64
import json
from pathlib import Path
import urllib.request
import websocket

parser = argparse.ArgumentParser()
parser.add_argument('--experiment',choices=['balanced','detail'],default='balanced')
args = parser.parse_args()
pages = json.load(urllib.request.urlopen('http://127.0.0.1:9231/json/list'))
page = next(p for p in pages if ('gnt-'+args.experiment+'.html') in p['url'])
ws = websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=120)
ws.send(json.dumps(dict(id=10,method='Page.enable')))
while json.loads(ws.recv()).get('id')!=10: pass
ws.send(json.dumps(dict(id=11,method='Page.reload',params=dict(ignoreCache=True))))
while json.loads(ws.recv()).get('method')!='Page.loadEventFired': pass
expression = '''(async()=>{
const check=(x,m)=>{if(!x)throw Error(m)};
let selections=0;
for(let i=0;i<cases.length;i++){
 selector.value=i;options();await load();const c=cases[i];
 for(const key of Object.keys(c.images)){
  left.value=key;draw();check(canvas.width===c.width,'Canvas resolution');
  check(Object.values(loaded).every(im=>im.naturalWidth===c.width),'Decoded dimensions');
  selections++;
 }
 check(c.kind!=='motion'||!c.images.photo,'Virtual camera has reference');
}
el('zoom').value=200;draw();check(canvas.style.width==='200%','Zoom');
el('wipe').value=25;draw();check(el('status').textContent.includes(cases[+selector.value].id),'Status');
el('play').click();check(timer!==null,'Path did not start');
await new Promise(resolve=>setTimeout(resolve,1200));
check(cases[+selector.value].id.startsWith('path_'),'Playback not on path');
el('play').click();check(timer===null,'Path did not pause');
selector.value=0;options();left.value=data.baseline_label||'adapt1800';right.value=data.candidate_label||'balanced5000';
el('zoom').value=100;el('wipe').value=50;await load();
return {status:'passed',cases:cases.length,selections,pathFrames:cases.filter(c=>c.kind==='motion').length,
 checks:['decode','dimensions','method choices','shared zoom','wipe','path playback and pause','reference availability']};
})()'''
ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
while True:
    result = json.loads(ws.recv())
    if result.get('id')==1: break
assert 'exceptionDetails' not in result['result'],result
value = result['result']['result']['value']
out = Path(__file__).resolve().parents[1]/'outputs'/('gnt_'+args.experiment)/'browser_qa.json'
out.write_text(json.dumps(value,indent=2))
print(json.dumps(value))
ws.send(json.dumps(dict(id=90,method='Page.captureScreenshot',params=dict(format='jpeg',quality=90))))
while True:
    screenshot = json.loads(ws.recv())
    if screenshot.get('id')==90: break
(out.parent/'browser.jpg').write_bytes(base64.b64decode(screenshot['result']['data']))
ws.send(json.dumps(dict(id=99,method='Browser.close')))
ws.close()
