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

layout(set=0,binding=1,std430) buffer States {vec4 data[];} states;
layout(set=0,binding=2,std430) buffer Incoming {float data[];} incoming;
layout(set=0,binding=3,std430) buffer Changes {int data[];} changes;
layout(set=0,binding=4,std430) buffer Heads {int data[];} heads;
layout(set=0,binding=5,std430) buffer Counts {int data[];} counts;
layout(set=0,binding=6,std430) buffer LiveDegrees {int data[];} live_degrees;
layout(set=0,binding=7,std430) buffer IncomingDegrees {int data[];} incoming_degrees;
layout(set=0,binding=8,std430) readonly buffer InitialIncoming {int data[];} initial_incoming;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;
 uint at=p.neuron_base+n;
 for(uint k=0u;k<3u;k++)states.data[3u*at+k]=vec4(0.0);
 incoming.data[at]=0.0;
 heads.data[at]=-1; counts.data[at]=0; live_degrees.data[at]=0;
 incoming_degrees.data[at]=initial_incoming.data[at];
 if(n==0u)for(uint k=0u;k<6u;k++)changes.data[6u*agent+k]=0;
}
