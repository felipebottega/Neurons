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
};
layout(set=0,binding=0,std430) readonly buffer ParameterBlock {Params data[];} parameters;
uint plane_size(Params p){return p.I*p.J;}

layout(set=0,binding=1,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=2,std430) readonly buffer Rescue_sourcesBuffer {int data[];} rescue_sources;
layout(set=0,binding=3,std430)  buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=4,std430)  buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=5,std430)  buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=6,std430)  buffer Live_degreesBuffer {int data[];} live_degrees;
layout(set=0,binding=7,std430)  buffer HeadsBuffer {int data[];} heads;
layout(set=0,binding=8,std430)  buffer LinksBuffer {int data[];} links;
layout(set=0,binding=9,std430)  buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=10,std430)  buffer Free_headsBuffer {int data[];} free_heads;
layout(set=0,binding=11,std430) readonly buffer Free_nextBuffer {int data[];} free_next;
layout(set=0,binding=12,std430)  buffer Free_countsBuffer {int data[];} free_counts;
int reserve_slot(uint agent,uint source,Params p){
 uint base=p.neuron_base,edgebase=p.edge_base;
 // Free-list nodes are immutable while homeostasis and structural run.
 // The only producer is the earlier prune pass; this prevents ABA during pop.
 int head=atomicAdd(free_heads.data[base+source],0);
 while(head>=0){
  int successor=free_next.data[edgebase+uint(head)*p.total+source];
  int previous=atomicCompSwap(free_heads.data[base+source],head,successor);
  if(previous==head){
   atomicAdd(free_counts.data[base+source],-1);
   return head;
  }
  head=previous;
 }
 int observed=atomicAdd(counts.data[base+source],0);
 while(observed<int(p.stride)){
  int previous=atomicCompSwap(counts.data[base+source],observed,observed+1);
  if(previous==observed){
   return observed;
  }
  observed=previous;
 }
 return -1;
}
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base; uint edgebase=p.edge_base;

 if(n>=p.total)return;
 if(roles.data[base+n]==1)return;
 int best=rescue_sources.data[base+n];if(best<0)return;
 uint source=uint(best);
 int pos=reserve_slot(agent,source,p);
 // No structural change is made at N_max. This is a normal saturation,
 // recorded separately from structural corruption in changes[3].
 if(pos<0){atomicAdd(changes.data[6u*agent+4u],1);return;}
 uint local_index=uint(pos)*p.total+source,at=edgebase+local_index;
 edges.data[at]=Edge(int(n),0.25*p.structural,0.0,0.0);
 links.data[at]=atomicExchange(heads.data[base+n],int(local_index));
 atomicAdd(incoming_degrees.data[base+n],1);atomicAdd(live_degrees.data[base+source],1);
 atomicAdd(changes.data[6u*agent+1u],1);

}
