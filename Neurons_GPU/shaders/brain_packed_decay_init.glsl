#[compute]
#version 450
// One-time GPU preprocessing of the phenotype-dependent decay coefficient.
// It does NOT depend on time, target, synaptic state or modulation.
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
layout(set=0,binding=1,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=2,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=3,std430) writeonly buffer EdgeConstantsBuffer {vec4 data[];} edge_constants;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;
 uint i=p.neuron_base+n;
 int role=roles.data[i];
 edge_constants.data[i]=vec4(role==1?0.0:exp(-32.0/phenotype.data[2u*i].y),
   float(role), phenotype.data[2u*i+1u].x, 0.0);
}
