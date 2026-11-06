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
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;
 uint at=p.neuron_base+n;
 for(uint k=0u;k<3u;k++)states.data[3u*at+k]=vec4(0.0);
 incoming.data[at]=0.0;
}
