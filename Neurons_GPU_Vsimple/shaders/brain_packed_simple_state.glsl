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

layout(set=0,binding=1,std430) readonly buffer Roles {int data[];} roles;
layout(set=0,binding=2,std430) buffer States {vec4 data[];} states;
layout(set=0,binding=3,std430) readonly buffer Incoming {float data[];} incoming;
layout(set=0,binding=4,std430) readonly buffer OutputReverse {int data[];} output_reverse;
layout(set=0,binding=5,std430) writeonly buffer Outputs {float data[];} outputs;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;
 uint at=p.neuron_base+n;
 vec4 a=states.data[3u*at];
 vec4 b=states.data[3u*at+1u];
 // The output of iteration t is the signal actually propagated in t,
 // not the signal queued for the next iteration.
 int output_channel=output_reverse.data[at];
 if(output_channel>=0)outputs.data[uint(output_channel)]=a.w;
 // The sum for iteration t+1 is reconstructed afresh. Input neurons
 // have no neural predecessors and do not retain their external sample.
 a.x=roles.data[at]==1?0.0:max(0.0,incoming.data[at]);
 b.w=incoming.data[at];
 states.data[3u*at]=a;
 states.data[3u*at+1u]=b;
}
