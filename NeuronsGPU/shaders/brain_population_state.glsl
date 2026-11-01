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

layout(set=0,binding=1,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=2,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=3,std430)  buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=4,std430) readonly buffer ModulationBuffer {vec2 data[];} modulation;
layout(set=0,binding=5,std430) readonly buffer IncomingBuffer {float data[];} incoming;
layout(set=0,binding=6,std430) readonly buffer OutputReverseBuffer {int data[];} output_reverse;
layout(set=0,binding=7,std430) writeonly buffer OutputStreamBuffer {float data[];} output_stream;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 if(n>=p.total)return;
 vec4 a=states.data[3u*(base+n)],b=states.data[3u*(base+n)+1u],c=states.data[3u*(base+n)+2u];
 // Previous output shader read the firing gate and normalized signal.
 // State updates a.x, b.w, c.x and c.z, but neither a.w nor c.w.
 // Thus writing here produces exactly the old post-plasticity output value.
 int output_channel=output_reverse.data[base+n];
 if(output_channel>=0){
   output_stream.data[uint(output_channel)]=c.w>0.5?a.w:0.0;
 }
 bool is_input=roles.data[base+n]==1;
 float retention=is_input?0.0:phenotype.data[2u*(base+n)].z;
 float v=max(0.0,retention*a.x+incoming.data[base+n]);
 if(!is_input){
    float act=a.z;
    if(act<0.05&&v<p.homeo_low)v+=(0.05-act)/0.05*(p.homeo_low-v);
    else if(act>0.95&&v>p.homeo_high)v-=(act-0.95)/0.05*(v-p.homeo_high);
 }
 a.x=v;b.w=incoming.data[base+n];c.x=modulation.data[base+n].x;c.z=modulation.data[base+n].y;
 states.data[3u*(base+n)]=a;states.data[3u*(base+n)+1u]=b;states.data[3u*(base+n)+2u]=c;

}
