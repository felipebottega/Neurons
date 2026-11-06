#[compute]
#version 450
// Mutable sparse ELLPACK: local edge index = slot * neuron_count + source.
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
struct Edge {int target; float weight; float trace; float padding;};
struct Params {
 uint I,J,K,total;
 uint stride,iteration; float delta,exponent;
 float refractory,mu,lambda_p,nu;
 float homeo_low,homeo_high,structural,input_gain;
 float input_floor; uint slice_k,initial_stride,selected_id;
 uint neuron_base, edge_base, initial_edge_base;
 uint radius_i, radius_j, radius_k, reserved0, reserved1;
 float fatigue_divisor; float decay_exponent; uint model_flags; uint reserved2;
};
layout(set=0,binding=0,std430) readonly buffer ParameterBlock {Params data[];} parameters;
uint plane_size(Params p){return p.I*p.J;}

layout(set=0,binding=1,std430) readonly buffer Offsets {uint data[];} offsets;
layout(set=0,binding=2,std430) readonly buffer Lengths {uint data[];} lengths;
layout(set=0,binding=3,std430) readonly buffer Indices {int data[];} indices;
layout(set=0,binding=4,std430) readonly buffer Edges {Edge data[];} edges;
layout(set=0,binding=5,std430) readonly buffer Emission {float data[];} emission;
layout(set=0,binding=6,std430) writeonly buffer Incoming {float data[];} incoming;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;
 uint base=p.neuron_base,eb=p.edge_base;
 // Sparse matrix-vector multiplication y = W_t q_t.
 // CSR stores inbound row slices, while ELLPACK owns mutable synaptic weights.
 // All reads occur BEFORE plasticity, preserving transmission timing.
 float sum=0.0;
 uint first=offsets.data[base+n];
 uint count=lengths.data[base+n];
 for(uint k=0u;k<count;k++){
   uint slot=uint(indices.data[eb+first+k]);
   uint source=slot%p.total;
   sum+=emission.data[base+source]*edges.data[eb+slot].weight;
 }
 incoming.data[base+n]=sum;
}
