#[compute]
#version 450
// Mutable sparse ELLPACK: local edge index = slot * neuron_count + source.
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

layout(set=0,binding=1,std430) readonly buffer Heads {int data[];} heads;
layout(set=0,binding=2,std430) readonly buffer Links {int data[];} links;
layout(set=0,binding=3,std430) readonly buffer Edges {Edge data[];} edges;
layout(set=0,binding=4,std430) readonly buffer Degrees {int data[];} degrees;
layout(set=0,binding=5,std430) readonly buffer Offsets {uint data[];} offsets;
layout(set=0,binding=6,std430) writeonly buffer Lengths {uint data[];} lengths;
layout(set=0,binding=7,std430) writeonly buffer Indices {int data[];} indices;
layout(set=0,binding=8,std430) readonly buffer Dirty {uint data[];} dirty;
layout(set=0,binding=9,std430) buffer Changes {int data[];} changes;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total || dirty.data[agent]==0u)return;
 uint base=p.neuron_base,eb=p.edge_base;
 int expected=degrees.data[base+n];
 if(expected<0 || uint(expected)>p.total*p.stride)return;
 uint first=offsets.data[base+n];
 if(first+uint(expected)>p.total*p.stride)return;
 uint written=0u;
 int at=heads.data[base+n];
 while(at>=0){
   uint local=uint(at);
   if(local>=p.total*p.stride || written>=uint(expected)){
     atomicAdd(changes.data[6u*agent+3u],1);
     break;
   }
   if(edges.data[eb+local].target!=int(n)){
     atomicAdd(changes.data[6u*agent+3u],1);
     break;
   }
   // CSR row contains edge slots, never materialized dense weights.
   indices.data[eb+first+written]=at;
   ++written;
   at=links.data[eb+local];
 }
 lengths.data[base+n]=written;
 if(written!=uint(expected))atomicAdd(changes.data[6u*agent+3u],1);
}
