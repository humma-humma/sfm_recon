let centers;
onmessage=({data})=>{
    if(data.centers){centers=new Float32Array(data.centers);return;}
    const {position:p,forward:f,sequence}=data;
    const depths=new Float32Array(centers.length/3);
    for(let i=0;i<depths.length;i++)depths[i]=(centers[3*i]-p[0])*f[0]+(centers[3*i+1]-p[1])*f[1]+(centers[3*i+2]-p[2])*f[2];
    const order=Uint32Array.from({length:depths.length},(_,i)=>i);
    order.sort((a,b)=>depths[b]-depths[a]);
    postMessage({order,sequence},[order.buffer]);
};
