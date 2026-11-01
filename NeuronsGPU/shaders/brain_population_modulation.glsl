#[compute]
#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
struct Edge {int target; float weight; float trace; float padding;};
struct Params {
 uint I,J,K,total;
 uint stride,iteration; float delta,exponent;
 float refractory,mu,lambda_p,nu;
 float homeo_low,homeo_high,structural,input_gain;
 float input_floor; uint slice_k,initial_stride,selected_id;
};
layout(set=0,binding=0,std430) readonly buffer ParameterBlock {Params data[];} parameters;
layout(push_constant,std430) uniform BrainIdentity {uint agent_id;} pc;
uint plane_size(Params p){return p.I*p.J;}
uint nb(Params p,uint agent){return 0u;}
uint eb(Params p,uint agent){return 0u;}

layout(set=0,binding=1,std430) readonly buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=2,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=3,std430) writeonly buffer ModulationBuffer {vec2 data[];} modulation;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 if(n>=p.total)return;
 uint x=n%p.I,y=(n/p.I)%p.J,z=n/plane_size(p);
 float sum=0.0; uint count=0u;
 if(x>0u){sum+=states.data[3u*(base+n-1u)+2u].x;count++;}
 if(x+1u<p.I){sum+=states.data[3u*(base+n+1u)+2u].x;count++;}
 if(y>0u){sum+=states.data[3u*(base+n-p.I)+2u].x;count++;}
 if(y+1u<p.J){sum+=states.data[3u*(base+n+p.I)+2u].x;count++;}
 if(z>0u){sum+=states.data[3u*(base+n-plane_size(p))+2u].x;count++;}
 if(z+1u<p.K){sum+=states.data[3u*(base+n+plane_size(p))+2u].x;count++;}
 float field=p.lambda_p*((1.0-p.nu)*states.data[3u*(base+n)+2u].x+p.nu*sum/float(max(1u,count)))+states.data[3u*(base+n)+2u].y;
 float effective=phenotype.data[2u*(base+n)+1u].z*(1.0-exp(-field));
 modulation.data[base+n]=vec2(field,effective);

}
