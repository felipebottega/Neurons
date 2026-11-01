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

layout(set=0,binding=1,std430) buffer HeadsBuffer {int data[];} heads;
layout(set=0,binding=2,std430) buffer LinksBuffer {int data[];} links;
layout(set=0,binding=3,std430) readonly buffer TransmissionsBuffer {vec2 data[];} transmissions;
layout(set=0,binding=4,std430)  buffer IncomingBuffer {float data[];} incoming;
layout(set=0,binding=5,std430) readonly buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=6,std430)  buffer FreeHeadsBuffer {int data[];} free_heads;
layout(set=0,binding=7,std430)  buffer FreeNextBuffer {int data[];} free_next;
layout(set=0,binding=8,std430)  buffer FreeCountsBuffer {int data[];} free_counts;
layout(set=0,binding=9,std430)  buffer DirtyBuffer {int data[];} dirty;
layout(set=0,binding=10,std430) buffer ChangesBuffer {int data[];} changes;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 if(n>=p.total)return;
 float accum=0.0;
 int at=heads.data[base+n];
 while(at>=0){
   vec2 transmit=transmissions.data[edgebase+uint(at)];
   if(int(transmit.x)==int(n))accum+=transmit.y;
   at=links.data[edgebase+uint(at)];
 }
 incoming.data[base+n]=accum;
 // This invocation exclusively owns the incoming list for destination n.
 // No other destination can reference its slots in a valid graph. After
 // gathering its current-step transmissions, it can unlink dead edges and
 // recycle their source slots without an extra dispatch or global barrier.
 if(dirty.data[base+n]!=0){
   int previous=-1;
   int slot=heads.data[base+n];
   while(slot>=0){
     uint current=uint(slot);
     if(current>=p.total*p.stride){atomicAdd(changes.data[3u],1);return;}
     int successor=links.data[current];
     int destination=edges.data[current].target;
     if(destination<0){
       if(previous<0)heads.data[base+n]=successor;
       else links.data[uint(previous)]=successor;
       uint source=current/p.stride;
       int local_slot=int(current-source*p.stride);
       int old_head=atomicExchange(free_heads.data[source],local_slot);
       free_next.data[current]=old_head;
       atomicAdd(free_counts.data[source],1);
       atomicAdd(changes.data[5u],1);
     }else{
       if(destination!=int(n)){atomicAdd(changes.data[3u],1);return;}
       previous=slot;
     }
     slot=successor;
   }
   dirty.data[base+n]=0;
 }

}
