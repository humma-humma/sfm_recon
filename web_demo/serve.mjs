// Serve only this demo directory, on loopback. No dependencies or external requests.
import http from 'node:http';
import {createReadStream,statSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(process.argv[2]||fileURLToPath(new URL('.',import.meta.url)));
const port=Number(process.env.PORT||8088);
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.jpg':'image/jpeg','.png':'image/png','.mp4':'video/mp4','.bin':'application/octet-stream'};
const server=http.createServer((req,res)=>{try{
    const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    const file=path.resolve(root,'.'+(pathname==='/'?'/index.html':pathname));
    const relative=path.relative(root,file);
    if(relative.startsWith('..')||path.isAbsolute(relative)){res.writeHead(403).end();return;}
    const info=statSync(file);
    if(!info.isFile()){res.writeHead(404).end();return;}
    const headers={'Content-Type':types[path.extname(file)]||'application/octet-stream',
        'Content-Length':info.size,'Cache-Control':'no-cache','X-Content-Type-Options':'nosniff','Accept-Ranges':'bytes'};
    let start=0,end=info.size-1,status=200;
    if(req.headers.range&&req.method==='GET'){
        const range=/^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
        if(range&&(range[1]||range[2])){
            start=range[1]?Number(range[1]):Math.max(0,info.size-Number(range[2]));
            end=range[1]&&range[2]?Math.min(Number(range[2]),end):end;
        }else{start=info.size;}
        if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start>end||start>=info.size){
            res.writeHead(416,{'Content-Range':`bytes */${info.size}`}).end();return;
        }
        status=206;headers['Content-Range']=`bytes ${start}-${end}/${info.size}`;
        headers['Content-Length']=end-start+1;
    }
    res.writeHead(status,headers);
    if(req.method==='HEAD'||info.size===0){res.end();return;}
    const stream=createReadStream(file,{start,end});
    stream.on('error',()=>res.destroy());stream.pipe(res);
}catch{res.writeHead(404).end('Not found');}});
server.listen(port,'127.0.0.1',()=>console.log(`Reconstruction Atlas: http://127.0.0.1:${port}\nCtrl+C to stop.`));
server.on('error',error=>{console.error(error.message);process.exitCode=1;});
