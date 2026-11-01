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
uint plane_size(Params p){return p.I*p.J;}
uint nb(Params p,uint agent){return 0u;}
uint eb(Params p,uint agent){return 0u;}
layout(push_constant,std430) uniform Init {uint agent_id;uint slice_id;uint original_stride;uint reserved;} pc;
layout(set=0,binding=1,std430) readonly buffer Source_edgesBuffer {Edge data[];} source_edges;
layout(set=0,binding=2,std430) readonly buffer Initial_countsBuffer {int data[];} initial_counts;
layout(set=0,binding=3,std430) writeonly buffer Initial_edgesBuffer {Edge data[];} initial_edges;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 
 uint who=pc.agent_id,local=n;
 if(local>=plane_size(p))return;
 uint source=pc.slice_id*plane_size(p)+local;
 uint initial=0u;
 uint at=source*p.initial_stride;
 int count=initial_counts.data[initial+source];
 for(int k=0;k<count;k++)initial_edges.data[at+uint(k)]=source_edges.data[local*pc.original_stride+uint(k)];

}
