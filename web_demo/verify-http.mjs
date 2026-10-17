// Start serve.mjs first. Read-only checks; never download the large model bodies.
import assert from 'node:assert/strict';
const base=process.env.DEMO_URL||'http://127.0.0.1:8088';
const manifestResponse=await fetch(`${base}/assets/manifest.json`);
assert.equal(manifestResponse.status,200);
const manifest=await manifestResponse.json();
let checked=0;
for(const file of ['index.html','app.js','style.css','sort-worker.js','research.html','assets/research/results.json']){
    const response=await fetch(`${base}/${file}`,{method:'HEAD'});
    assert.equal(response.status,200,file);
    assert(Number(response.headers.get('content-length'))>0,file);
    checked++;
}
for(const name of ['box','boot','milk','scene']){
    for(const [suffix,size] of [['bin',manifest[name].bytes],['sh.bin',manifest[name].sh_bytes]]){
        const response=await fetch(`${base}/assets/${name}.${suffix}`,{method:'HEAD'});
        assert.equal(response.status,200);
        assert.equal(Number(response.headers.get('content-length')),size);
        checked++;
    }
}
const denied=await fetch(`${base}/%2e%2e%2fREADME.md`,{method:'HEAD'});
assert([403,404].includes(denied.status),'Server must not serve files outside demo');
const original=Buffer.from(await(await fetch(`${base}/app.js`)).arrayBuffer());
for(const [range,start,end] of [['bytes=0-15',0,15],['bytes=16-',16,original.length-1],
    ['bytes=-16',original.length-16,original.length-1],['bytes=0-999999999',0,original.length-1]]){
    const response=await fetch(`${base}/app.js`,{headers:{Range:range}});
    assert.equal(response.status,206,range);
    assert.equal(response.headers.get('content-range'),`bytes ${start}-${end}/${original.length}`);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()),original.subarray(start,end+1));
}
for(const range of [`bytes=${original.length}-`,'bytes=16-2','bytes=-0','bytes=0-1,3-4']){
    const response=await fetch(`${base}/app.js`,{headers:{Range:range}});
    assert.equal(response.status,416,range);
    assert.equal(response.headers.get('content-range'),`bytes */${original.length}`);
}
console.log(JSON.stringify({status:'pass',http_resources:checked,outside_root_denied:true,
    byte_range_checks:8,browser_interaction_tested:false}));
