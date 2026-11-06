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
 uint neuron_base, edge_base, initial_edge_base;
 uint radius_i, radius_j, radius_k, reserved0, reserved1;
};
layout(set=0,binding=0,std430) readonly buffer ParameterBlock {Params data[];} parameters;
uint plane_size(Params p){return p.I*p.J;}

layout(set=0,binding=1,std430) readonly buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=2,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=3,std430) writeonly buffer ModulationBuffer {vec2 data[];} modulation;
layout(set=0,binding=4,std430) readonly buffer RolesBuffer {int data[];} roles;
// SourceRuntime.x: emission q_t; .y: normalized signal s_t; .z: fired gate;
// .w: activity trace R_t. These are immutable during fused7/state_select7.
layout(set=0,binding=5,std430) writeonly buffer SourceRuntimeBuffer {vec4 data[];} source_runtime;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base; uint edgebase=p.edge_base;

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
 // The sparse matvec uses the pre-plasticity emission vector q_t.
 // Emission is fixed for this step, regardless of subsequent edge deaths.
 vec4 s=states.data[3u*(base+n)];
 vec4 c=states.data[3u*(base+n)+2u];
 float source_gain=phenotype.data[2u*(base+n)].w;
 bool fired=c.w>0.5;
 float q=fired?source_gain*(roles.data[base+n]==1?p.input_gain*s.x:s.w):0.0;
 source_runtime.data[base+n]=vec4(q,s.w,c.w,s.z);

}
