#[compute]
#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
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

layout(set=0,binding=1,std430) readonly buffer Phenotype {vec4 data[];} phenotype;
layout(set=0,binding=2,std430) readonly buffer Roles {int data[];} roles;
layout(set=0,binding=3,std430) buffer States {vec4 data[];} states;
layout(set=0,binding=4,std430) readonly buffer Inputs {float data[];} inputs;
layout(set=0,binding=5,std430) readonly buffer InputReverse {int data[];} input_reverse;
layout(set=0,binding=6,std430) writeonly buffer Emission {float data[];} emission;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;
 uint at=p.neuron_base+n;
 vec4 a=states.data[3u*at];
 vec4 b=states.data[3u*at+1u];
 vec4 c=states.data[3u*at+2u];
 bool is_input=roles.data[at]==1;
 // External input is injected ONLY via the designated input channel.
 // No state from a previous input sample survives this step.
 if(is_input){
    int channel=input_reverse.data[at];
    a.x=channel>=0?inputs.data[uint(channel)]:0.0;
 }
 // A neuron's state is exactly its previous step's incoming sum (or
 // the current external sample for an input). No retention or threshold.
 float s=max(0.0,a.x);
 float normalized=s>0.0?(1.0-exp(-s/p.exponent)):0.0;
 a.w=normalized;
 b.x=s; b.y=0.0; b.z=0.0;
 a.y=0.0; a.z=0.0;
 c=vec4(0.0,0.0,0.0,s>0.0?1.0:0.0);
 float q=is_input?p.input_gain*s:phenotype.data[2u*at].w*normalized;
 emission.data[at]=q;
 states.data[3u*at]=a;
 states.data[3u*at+1u]=b;
 states.data[3u*at+2u]=c;
}
