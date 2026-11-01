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
layout(push_constant,std430) uniform Pass {uint stage;uint agent_id;uint preflight_offset;} pc;
layout(set=0,binding=1,std430) readonly buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=2,std430) readonly buffer Rescue_sourcesBuffer {int data[];} rescue_sources;
layout(set=0,binding=3,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=4,std430)  buffer DemandBuffer {int data[];} demand;
layout(set=0,binding=5,std430)  buffer RequiredBuffer {int data[];} required;
layout(set=0,binding=6,std430)  buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=7,std430)  buffer StatusBuffer {int data[];} status;
layout(set=0,binding=8,std430) readonly buffer Free_countsBuffer {int data[];} free_counts;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 if(n>=p.total)return;
 if(pc.stage==0u){
   demand.data[base+n]=0;
   if(n==0u)required.data[agent]=int(p.stride);
 }else if(pc.stage==1u){
   // Stage 0 is fused into select. Stage 1 is always dispatched before
   // stage 2, with a compute barrier between them.
   if(n==0u)status.data[pc.preflight_offset+agent]=0;
   int src=rescue_sources.data[base+n];
   if(src>=0&&uint(src)<p.total)atomicAdd(demand.data[base+uint(src)],1);
 }else if(pc.stage==2u){
   int structural_maximum=roles.data[base+n]==0?1:0;
   int available=free_counts.data[base+n];
   int occupied=counts.data[base+n];
   if(available<0||available>occupied){
     atomicAdd(changes.data[3u],1);
     // Every offending invocation contributes directly to the preflight flag.
     atomicOr(status.data[pc.preflight_offset+agent],1);
   }
   int new_slots=max(0,demand.data[base+n]+structural_maximum-available);
   atomicMax(required.data[agent],counts.data[base+n]+new_slots);
 }
 // Stage 2 includes errors from earlier phases (e.g. gather) in the
 // preflight flag. AtomicAdd(0) avoids a non-atomic read racing the stage-2
 // atomic additions; the other invocations also atomically OR their errors.
 if(n==0u && pc.stage==2u)
   atomicOr(status.data[pc.preflight_offset+agent],atomicAdd(changes.data[3u],0));
 // Stage 3 snapshots only the six counters needed for integrity and Viewer.
 if(n==0u && pc.stage==3u)for(uint m=0u;m<6u;m++)status.data[6u*agent+m]=changes.data[m];

}
