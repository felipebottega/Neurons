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
};
layout(set=0,binding=0,std430) readonly buffer ParameterBlock {Params data[];} parameters;
uint plane_size(Params p){return p.I*p.J;}

// During reset, iteration is zero. All later iterations detect topology
// changes from the counters, so no push constants are necessary.
layout(set=0,binding=1,std430) readonly buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=2,std430) buffer CursorBuffer {uint data[];} cursor;
layout(set=0,binding=3,std430) buffer DirtyBuffer {uint data[];} dirty;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 if(gl_GlobalInvocationID.x!=0u)return;
 // Mutation counts were produced after this step's propagation.
 // If the sparse graph has not changed, keep the existing CSR representation.
 uint changed=(parameters.data[agent].iteration==0u)?1u:0u;
 if(changes.data[6u*agent+0u]!=0 || changes.data[6u*agent+1u]!=0 || changes.data[6u*agent+2u]!=0)
   changed=1u;
 dirty.data[agent]=changed;
 if(changed!=0u)cursor.data[agent]=0u;
}
