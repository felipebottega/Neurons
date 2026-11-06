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

layout(set=0,binding=1,std430) readonly buffer Degrees {int data[];} degrees;
layout(set=0,binding=2,std430) buffer CursorBuffer {uint data[];} cursor;
layout(set=0,binding=3,std430) readonly buffer DirtyBuffer {uint data[];} dirty;
layout(set=0,binding=4,std430) writeonly buffer Offsets {uint data[];} offsets;
layout(set=0,binding=5,std430) buffer Changes {int data[];} changes;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total || dirty.data[agent]==0u)return;
 int degree=degrees.data[p.neuron_base+n];
 if(degree<0){atomicAdd(changes.data[6u*agent+3u],1);return;}
 uint first=atomicAdd(cursor.data[agent],uint(degree));
 offsets.data[p.neuron_base+n]=first;
 if(first+uint(degree)>p.total*p.stride)
   atomicAdd(changes.data[6u*agent+3u],1);
}
