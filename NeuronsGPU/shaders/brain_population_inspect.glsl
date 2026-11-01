#[compute]
#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
struct Edge {int target; float weight; float trace; float padding;};
struct Params {uint I,J,K,total;uint stride,iteration;float delta,exponent;
 float refractory,mu,lambda_p,nu;float homeo_low,homeo_high,structural,input_gain;
 float input_floor;uint slice_k,initial_stride,selected_id;};
layout(set=0,binding=0,std430) readonly buffer Parameters {Params data[];} parameters;
layout(push_constant,std430) uniform BrainIdentity {uint agent_id;} pc;
layout(set=0,binding=1,std430) readonly buffer Edges {Edge data[];} edges;
layout(set=0,binding=2,std430) readonly buffer Counts {int data[];} counts;
layout(set=0,binding=3,std430) writeonly buffer Stats {vec2 data[];} stats;
void main(){
 uint n=gl_GlobalInvocationID.x;Params p=parameters.data[pc.agent_id];
 if(n>=p.total)return;
 vec2 sum=vec2(0.0);
 for(int k=0;k<counts.data[n];k++){
  Edge e=edges.data[n*p.stride+uint(k)];
  if(e.target>=0){sum.x+=e.weight;sum.y+=e.trace;}
 }
 stats.data[n]=sum;
}
