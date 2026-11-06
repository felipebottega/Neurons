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
 float fatigue_divisor; float decay_exponent; uint model_flags; uint reserved2;
};
layout(set=0,binding=0,std430) readonly buffer ParameterBlock {Params data[];} parameters;
uint plane_size(Params p){return p.I*p.J;}

layout(set=0,binding=1,std430) readonly buffer Initial_edgesBuffer {Edge data[];} initial_edges;
layout(set=0,binding=2,std430) readonly buffer Initial_countsBuffer {int data[];} initial_counts;
layout(set=0,binding=3,std430)  buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=4,std430)  buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=5,std430)  buffer Live_degreesBuffer {int data[];} live_degrees;
layout(set=0,binding=6,std430)  buffer HeadsBuffer {int data[];} heads;
layout(set=0,binding=7,std430)  buffer LinksBuffer {int data[];} links;
layout(set=0,binding=8,std430)  buffer ChangesBuffer {int data[];} changes;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base; uint edgebase=p.edge_base;

 if(n>=p.total)return;
 int count=initial_counts.data[base+n];
 counts.data[base+n]=count;live_degrees.data[base+n]=count;
 uint start=edgebase+n;
 for(int k=0;k<count;k++){
   uint local_index=uint(k)*p.total+n;
   uint initial_offset=p.initial_edge_base+n*p.initial_stride+uint(k);
   Edge e=initial_edges.data[initial_offset];
   edges.data[edgebase+local_index]=e;
   if(e.target<0||uint(e.target)>=p.total){atomicAdd(changes.data[6u*agent+3u],1);links.data[edgebase+local_index]=-1;continue;}
   links.data[edgebase+local_index]=atomicExchange(heads.data[base+uint(e.target)],int(local_index));
 }
 // All never-used slots are explicitly marked absent. A dead edge is also
 // marked by target=-1 and its slot is recyclable by the per-source free list.
 for(int k=count;k<int(p.stride);++k){
   uint local_index=uint(k)*p.total+n;
   edges.data[edgebase+local_index]=Edge(-1,0.0,0.0,0.0);
   links.data[edgebase+local_index]=-1;
 }

}
