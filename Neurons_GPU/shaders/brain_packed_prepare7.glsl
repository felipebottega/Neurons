#[compute]
#version 450
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

layout(set=0,binding=1,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=2,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=3,std430)  buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=4,std430) readonly buffer InputStreamBuffer {float data[];} input_stream;
layout(set=0,binding=5,std430)  buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=6,std430) readonly buffer InputReverseBuffer {int data[];} input_reverse;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base; uint edgebase=p.edge_base;

 if(n>=p.total)return;
 // Incoming is overwritten for all neurons by fused7 every step.
 if(n==0u)for(uint m=0u;m<6u;m++)changes.data[6u*agent+m]=0;
 vec4 a=states.data[3u*(base+n)];
 vec4 b=states.data[3u*(base+n)+1u];
 vec4 c=states.data[3u*(base+n)+2u];
 vec4 phenotype0=phenotype.data[2u*(base+n)];
 vec4 phenotype1=phenotype.data[2u*(base+n)+1u];
 bool is_input=roles.data[base+n]==1;
 if(is_input){
   int channel=input_reverse.data[base+n];
   if(channel>=0)a.x=input_stream.data[uint(channel)];
 }
 b.x=a.x;b.y=a.y;b.z=is_input?0.0:phenotype0.x*(1.0+a.y);
 if(!is_input)a.z*=p.mu;
 a.w=a.x>0.0?1.0-exp(-a.x/p.exponent):0.0;
 if(!is_input && a.x>0.0)a.z+=(1.0-p.mu)*a.w;
 bool fired=(a.x>0.0)&&(is_input || a.w>b.z);
 c.w=fired?1.0:0.0;
 c.y=fired?phenotype1.y*a.w:0.0;
 if(!is_input)a.y=fired?p.refractory:0.5*a.y;
 states.data[3u*(base+n)]=a;
 states.data[3u*(base+n)+1u]=b;
 states.data[3u*(base+n)+2u]=c;

}
