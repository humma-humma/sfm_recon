"""Exercise the comparison in an isolated Chrome CDP session on port 9231."""
import json
from pathlib import Path
import urllib.request
import websocket

pages = json.load(urllib.request.urlopen('http://127.0.0.1:9231/json/list'))
page = next(p for p in pages if 'gnt-tuning.html' in p['url'])
ws = websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=120)
ws.send(json.dumps(dict(id=10,method='Page.enable')))
while json.loads(ws.recv()).get('id')!=10: pass
ws.send(json.dumps(dict(id=11,method='Page.reload',params=dict(ignoreCache=True))))
loaded = False
while not loaded:
    message = json.loads(ws.recv())
    loaded = message.get('method')=='Page.loadEventFired'
expression = '''(async()=>{
let selections=0,viewsChecked=0;
const check=(value,message)=>{if(!value)throw Error(message)};
for(const [dataset,cases] of Object.entries(data.cases)){
 controls.dataset.value=dataset;views();
 for(const c of cases){
  controls.view.value=c.id;methods();await load();
  check(images!==null,`Failed initial load ${dataset}/${c.id}: ${controls.status.textContent}`);
  check(c.urls[controls.left.value]&&c.urls[controls.right.value],'Missing default method');
  for(const method of Object.keys(c.urls)){
   const right=controls.right.value;controls.left.value=method;await load();
   check(images!==null,'Failed method '+method);
   check(controls.right.value===right,'Left choice changed right');
   check(images.every(i=>i.naturalWidth===c.width&&i.naturalHeight===c.height),'Image dimensions');
   selections++;
  }
  controls.zoom.value='2';draw();check(controls.status.textContent.includes('2× crop zoom'),'Crop zoom');
  document.getElementById('toggle').click();check(['0','100'].includes(controls.wipe.value),'Toggle');
  document.getElementById('reset').click();check(controls.zoom.value==='1'&&center.every(x=>x===.5),'Reset');
  check(c.kind!=='motion'||!c.urls.gt,'Motion incorrectly has reference');
  viewsChecked++;
 }
}
return {status:'passed',views:viewsChecked,selections,checks:['decoded matching images','available-method fallback','independent selections','crop zoom','toggle/reset','reference availability']};
})()'''
ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
while True:
    result = json.loads(ws.recv())
    if result.get('id')==1: break
ws.close()
assert 'exceptionDetails' not in result['result'],result
value = result['result']['result']['value']
out = Path(__file__).resolve().parents[1]/'outputs/gnt_tuning/browser_qa.json'
out.write_text(json.dumps(value,indent=2))
print(json.dumps(value))
