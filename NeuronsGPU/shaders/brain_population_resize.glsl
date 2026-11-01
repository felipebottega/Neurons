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
layout(push_constant,std430) uniform Resize {uint new_stride;uint agent_id;} pc;
layout(set=0,binding=1,std430) readonly buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=2,std430)  buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=3,std430)  buffer HeadsBuffer {int data[];} heads;
layout(set=0,binding=4,std430) writeonly buffer New_edgesBuffer {Edge data[];} new_edges;
layout(set=0,binding=5,std430) writeonly buffer New_linksBuffer {int data[];} new_links;
layout(set=0,binding=6,std430)  buffer Free_headsBuffer {int data[];} free_heads;
layout(set=0,binding=7,std430)  buffer Free_countsBuffer {int data[];} free_counts;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 if(n>=p.total)return;
 uint src=edgebase+n*p.stride;
 uint dst_base=n*pc.new_stride;
 int count=counts.data[base+n];
 uint packed=0u;
 for(int k=0;k<count;k++){
   Edge e=edges.data[src+uint(k)];
   if(e.target<0)continue;
   uint local_index=dst_base+packed;
   new_edges.data[local_index]=e;
   new_links.data[local_index]=atomicExchange(heads.data[base+uint(e.target)],int(local_index));
   packed++;
 }
 counts.data[base+n]=int(packed);
 free_heads.data[base+n]=-1;
 free_counts.data[base+n]=0;

}
