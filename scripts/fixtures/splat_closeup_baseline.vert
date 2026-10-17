#version 300 es
precision highp float; precision highp int;
layout(location=0) in uint splatIndex;
uniform sampler2D splats; uniform vec3 eye,right,up,forward; uniform vec2 viewport; uniform float focal;
out vec2 local; out vec4 tint;
vec4 readAt(int i){return texelFetch(splats,ivec2(i%2048,i/2048),0);}
void main(){int b=int(splatIndex)*4;vec4 p=readAt(b);vec4 c=readAt(b+1);vec4 a=readAt(b+2);vec4 d=readAt(b+3);
vec3 delta=p.xyz-eye; vec3 v=vec3(dot(delta,right),dot(delta,up),dot(delta,forward));
if(v.z<0.005){gl_Position=vec4(2.,2.,2.,1.);local=vec2(9.);tint=vec4(0.);return;}
mat3 cov=mat3(a.x,a.y,a.z,a.y,a.w,d.x,a.z,d.x,d.y);
vec3 jx=focal/v.z*(right-forward*v.x/v.z),jy=focal/v.z*(up-forward*v.y/v.z);
float xx=dot(jx,cov*jx)+.3,xy=dot(jx,cov*jy),yy=dot(jy,cov*jy)+.3;
float mid=.5*(xx+yy),gap=sqrt(max(0.,.25*(xx-yy)*(xx-yy)+xy*xy));
vec2 axis=abs(xy)>1e-7?normalize(vec2(xy,mid+gap-xx)):(xx>=yy?vec2(1.,0.):vec2(0.,1.));
vec2 major=axis*min(sqrt(max(.1,mid+gap)),2048.),minor=vec2(-axis.y,axis.x)*min(sqrt(max(.1,mid-gap)),2048.);
vec2 corner=vec2((gl_VertexID==1||gl_VertexID==3)?1.:-1.,gl_VertexID>=2?1.:-1.);local=corner*3.;
vec2 screen=focal*v.xy/v.z+3.*(major*corner.x+minor*corner.y);
gl_Position=vec4(2.*screen/viewport,0.,1.);tint=vec4(c.rgb,p.w);}