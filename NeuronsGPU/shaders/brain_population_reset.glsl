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

layout(set=0,binding=1,std430)  buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=2,std430)  buffer ModulationBuffer {vec2 data[];} modulation;
layout(set=0,binding=3,std430)  buffer IncomingBuffer {float data[];} incoming;
layout(set=0,binding=4,std430)  buffer PendingBuffer {float data[];} pending;
layout(set=0,binding=5,std430)  buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=6,std430)  buffer Rescue_flagsBuffer {int data[];} rescue_flags;
layout(set=0,binding=7,std430)  buffer Rescue_sourcesBuffer {int data[];} rescue_sources;
layout(set=0,binding=8,std430)  buffer HeadsBuffer {int data[];} heads;
layout(set=0,binding=9,std430)  buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=10,std430)  buffer Live_degreesBuffer {int data[];} live_degrees;
layout(set=0,binding=11,std430)  buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=12,std430) readonly buffer Initial_incomingBuffer {int data[];} initial_incoming;
layout(set=0,binding=13,std430)  buffer Free_headsBuffer {int data[];} free_heads;
layout(set=0,binding=14,std430)  buffer Free_countsBuffer {int data[];} free_counts;
layout(set=0,binding=15,std430)  buffer DirtyBuffer {int data[];} dirty;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 if(n>=p.total)return;
 for(uint a=0u;a<3u;a++)states.data[3u*(base+n)+a]=vec4(0.0);
 modulation.data[base+n]=vec2(0.0);
 incoming.data[base+n]=0.0;pending.data[base+n]=0.0;
 rescue_flags.data[base+n]=0;rescue_sources.data[base+n]=0;
 heads.data[base+n]=-1;counts.data[base+n]=0;live_degrees.data[base+n]=0;
 free_heads.data[base+n]=-1;free_counts.data[base+n]=0;dirty.data[base+n]=0;
 incoming_degrees.data[base+n]=initial_incoming.data[base+n];
 if(n==0u)for(uint m=0u;m<6u;m++)changes.data[m]=0;

}
