"""Check all comparison frames decode and controls show the matching sources."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket
import argparse

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--balanced',action='store_true');args=parser.parse_args()
name='gnt-balanced-supervision.html' if args.balanced else 'gnt-dense-supervision.html'
port=9235 if args.balanced else 9234
profile='balanced_supervision_qa' if args.balanced else 'dense_supervision_qa'
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new',
    '--disable-gpu','--window-size=1440,1000',f'--remote-debugging-port={port}',
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
    const check=(x,m)=>{if(!x)throw Error(m)};
    const g=document.getElementById('group'),f=document.getElementById('frame'),o=document.getElementById('order');
    check(document.querySelectorAll('table tr').length===__METRIC_ROWS__,'Metric rows');
    check([...document.images].map(i=>i.id).join(',')==='__PANELS__','Panel identities and order');
    check([...document.querySelectorAll('figcaption')].map(x=>x.textContent).join('|')==='__LABELS__','Panel labels');
    let frames=0;
    const report=await fetch('assets/gnt_finalization/gnt_dense_supervision/results.json').then(r=>r.json());
    for(const [group,count] of [['withheld',84],['adaptive',report.plan.adaptive.length],['added',85],['original_training',275]]){
      g.value=group;g.dispatchEvent(new Event('change'));check(f.options.length===count,'Split size');
      for(let i=0;i<count;i++){
        f.value=i;f.dispatchEvent(new Event('change'));
        const chosen=data.find(r=>r.group===group&&r.index===+f.selectedOptions[0].textContent.split(' ')[1]);
        for(const image of document.images){await image.decode();check(image.naturalWidth===960&&image.naturalHeight===540,'Dimensions');check(image.src===image.parentElement.href,'Link');check(image.src.endsWith(image.id==='target'?chosen.target:chosen[image.id].file),'Matched source');}
        frames++;
      }
    }
    g.value='withheld';g.dispatchEvent(new Event('change'));
    for(const order of ['best','worst','index']){o.value=order;o.dispatchEvent(new Event('change'));for(const image of document.images)await image.decode();
      const sorted=data.filter(r=>r.group==='withheld').sort((a,b)=>order==='best'?b.delta-a.delta:order==='worst'?a.delta-b.delta:a.index-b.index);
      check(f.selectedOptions[0].textContent==='Frame '+sorted[0].index,'Sort order');}
    document.getElementById('next').click();check(f.value==='1','Next');
    document.getElementById('prev').click();check(f.value==='0','Previous');
    return {status:'passed',frames,images:frames*document.images.length,panels:document.images.length,checks:['panel identities/order/labels','all image decodes','dimensions','matching sources and full-size links','split counts','sorting','next/previous']};})()'''
    expression=expression.replace('__METRIC_ROWS__','0' if args.balanced else '16')
    expression=expression.replace('__PANELS__','target,dense,balanced90' if args.balanced else 'target,initial,control,dense')
    expression=expression.replace('__LABELS__','1. Best GNT — quality reference|2. Previous test — uniform dense supervision|3. Current test — balanced supervision (90% original / 10% added)' if args.balanced else 'GNT target|Initial GNT-only 10k|Control: original targets, +2k|Dense: +85 targets, +2k')
    ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
    while True:
        result=json.loads(ws.recv())
        if result.get('id')==1:break
    assert 'exceptionDetails' not in result['result'],result
    value=result['result']['result']['value']
    output='balanced_supervision_browser_qa.json' if args.balanced else 'dense_supervision_browser_qa.json'
    (ROOT/'outputs/gnt_finalization'/output).write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)
finally:
    ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
