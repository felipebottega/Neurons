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

layout(set=0,binding=1,std430) buffer HeadsBuffer {int data[];} heads;
layout(set=0,binding=2,std430) buffer LinksBuffer {int data[];} links;
layout(set=0,binding=3,std430) readonly buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=4,std430) buffer FreeHeadsBuffer {int data[];} free_heads;
layout(set=0,binding=5,std430) buffer FreeNextBuffer {int data[];} free_next;
layout(set=0,binding=6,std430) buffer FreeCountsBuffer {int data[];} free_counts;
layout(set=0,binding=7,std430) buffer DirtyBuffer {int data[];} dirty;
layout(set=0,binding=8,std430) buffer ChangesBuffer {int data[];} changes;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;
 uint base=p.neuron_base,eb=p.edge_base;
 // No traversal on unchanged rows. Recycle deaths BEFORE new connections.
 if(dirty.data[base+n]==0)return;
 int previous=-1;
 int slot=heads.data[base+n];
 uint steps=0u;
 while(slot>=0){
   uint local=uint(slot);
   if(local>=p.total*p.stride || steps++>=p.total*p.stride){
     atomicAdd(changes.data[6u*agent+3u],1);
     return;
   }
   int successor=links.data[eb+local];
   int destination=edges.data[eb+local].target;
   if(destination<0){
     if(previous<0)heads.data[base+n]=successor;
     else links.data[eb+uint(previous)]=successor;
     uint source=local%p.total;
     int source_slot=int(local/p.total);
     int old_head=atomicExchange(free_heads.data[base+source],source_slot);
     free_next.data[eb+local]=old_head;
     atomicAdd(free_counts.data[base+source],1);
     atomicAdd(changes.data[6u*agent+5u],1);
   }else{
     if(destination!=int(n)){
       atomicAdd(changes.data[6u*agent+3u],1);
       return;
     }
     previous=slot;
   }
   slot=successor;
 }
 dirty.data[base+n]=0;
}
