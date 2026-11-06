#[compute]
#version 450
// Fixed-size synaptic graph: snapshot six production GPU counters once per iteration.
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
layout(set=0,binding=1,std430) readonly buffer Changes {int data[];} changes;
layout(set=0,binding=2,std430) writeonly buffer Status {int data[];} status;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 if(gl_GlobalInvocationID.x!=0u || parameters.data[agent].total==0u)return;
 for(uint m=0u;m<6u;++m)
  status.data[6u*agent+m]=changes.data[6u*agent+m];
}
