"""Check diagnostic image delivery and native/black background switching."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import websocket

ROOT=Path(__file__).resolve().parents[1]
subprocess.Popen([r'C:\Program Files\Google\Chrome\Application\chrome.exe','--headless=new',
    '--disable-gpu','--window-size=1440,1000','--remote-debugging-port=9233',
    r'--user-data-dir=C:\Users\mopu01\AppData\Local\Temp\sfm_lightweight_methods_v1\transfer_browser_qa',
    'http://localhost:8088/gnt-transfer-diagnosis.html'],creationflags=subprocess.CREATE_NO_WINDOW,
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
deadline=time.monotonic()+30
while True:
    try:
        pages=json.load(urllib.request.urlopen('http://127.0.0.1:9233/json/list'))
        page=next(p for p in pages if 'gnt-transfer-diagnosis.html' in p['url']);break
    except (OSError,StopIteration):
        if time.monotonic()>deadline:raise
        time.sleep(.5)
ws=websocket.create_connection(page['webSocketDebuggerUrl'],suppress_origin=True,timeout=120)
try:
    ws.send(json.dumps(dict(id=10,method='Page.enable')))
    while json.loads(ws.recv()).get('id')!=10:pass
    ws.send(json.dumps(dict(id=11,method='Page.reload',params=dict(ignoreCache=True))))
    while json.loads(ws.recv()).get('method')!='Page.loadEventFired':pass
    expression='''(async()=>{
    const check=(x,m)=>{if(!x)throw Error(m)};
    const images=[...document.querySelectorAll('img')];
    const fits=images.filter(i=>i.classList.contains('fit'));
    const targets=images.filter(i=>!i.classList.contains('fit'));
    check(images.length>=48&&fits.length===targets.length,'Image counts');
    check(document.querySelectorAll('table tr').length===7,'Metric rows');
    for(const i of images){i.loading='eager';await i.decode();check(i.naturalWidth===960&&i.naturalHeight===540,'Dimensions');}
    const originals=targets.map(i=>i.src);
    const mode=document.getElementById('mode');
    for(const value of ['black','native']){
      mode.value=value;mode.dispatchEvent(new Event('change'));
      for(const i of fits){await i.decode();check(i.src.endsWith('_'+value+'.png'),'Background source');check(i.parentElement.href===i.src,'Full-size link');}
      check(targets.every((i,j)=>i.src===originals[j]),'Targets changed');
    }
    return {status:'passed',images:images.length,background_switches:2,checks:['image decoding','dimensions','metric rows','background controls','full-size links','unchanged targets']};})()'''
    ws.send(json.dumps(dict(id=1,method='Runtime.evaluate',params=dict(expression=expression,awaitPromise=True,returnByValue=True))))
    while True:
        result=json.loads(ws.recv())
        if result.get('id')==1:break
    assert 'exceptionDetails' not in result['result'],result
    value=result['result']['result']['value']
    (ROOT/'outputs/gnt_finalization/transfer_browser_qa.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)
finally:
    ws.send(json.dumps(dict(id=99,method='Browser.close')));ws.close()
