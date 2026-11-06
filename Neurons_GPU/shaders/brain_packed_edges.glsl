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

layout(set=0,binding=1,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=2,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=3,std430) readonly buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=4,std430) readonly buffer ModulationBuffer {vec2 data[];} modulation;
layout(set=0,binding=5,std430) buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=6,std430) readonly buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=7,std430) buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=8,std430) buffer Live_degreesBuffer {int data[];} live_degrees;
layout(set=0,binding=9,std430) buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=10,std430) buffer DirtyBuffer {int data[];} dirty;
// Runtime Edge.padding stores signed decay rounding debt (weight - debt).
// precise prevents reassociation from optimizing the compensation away.
float subtract_decay(float weight, float decrement, inout float residual) {
    precise float adjusted = decrement + residual;
    precise float next = weight - adjusted;
    precise float applied = weight - next;
    precise float remainder = adjusted - applied;
    residual = remainder;
    if (next <= 0.0) {
        residual = 0.0;
        return 0.0;
    }
    return next;
}

void main(){
 uint agent=gl_GlobalInvocationID.z;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base, edgebase=p.edge_base;
 if(n>=p.total)return;

 // Each invocation owns a vector tile of four ELLPACK edge slots.
 // Slots are independent, and neighboring lanes process neighboring sources.
 int count=counts.data[base+n];
 uint tile_first=4u*gl_GlobalInvocationID.y;
 if(tile_first>=uint(count))return;

 vec4 phenotype0=phenotype.data[2u*(base+n)],phenotype1=phenotype.data[2u*(base+n)+1u];
 vec4 a=states.data[3u*(base+n)],c=states.data[3u*(base+n)+2u];
 bool is_input=roles.data[base+n]==1;
 bool fired=c.w>0.5;
 float decay=is_input?0.0:exp(-32.0/phenotype0.y);
 float hebbian=phenotype1.x;
 uint start=edgebase+n;
 int deaths=0;

 for(uint k=tile_first;k<min(tile_first+4u,uint(count));k++){
   uint index=start+k*p.total;
   // Avoid fetching the other 12 bytes of a dead edge. Dead slots are not
   // in incoming lists after gather and need no repeated null transmission.
   int target_id=edges.data[index].target;
   if(target_id<0 || uint(target_id)>=p.total)continue;
   uint target=uint(target_id);
   Edge e=edges.data[index];

   float eligibility=p.mu*e.trace;
   float weight=e.weight;
   // The target role and state do not change during the edges stage.
   // Reuse one vec4 when both Hebbian eligibility and output attenuation
   // need the target state, without loading it for silent non-output edges.
   int target_role=is_input?0:roles.data[base+target];
   bool need_target_state=fired || target_role==2;
   vec4 target_state=vec4(0.0);
   if(need_target_state)target_state=states.data[3u*(base+target)];
   if(fired){
     weight+=hebbian*(1.0-weight);
     if(!is_input)e.padding*=1.0-hebbian;
     eligibility+=(1.0-p.mu)*a.w*target_state.w;
   }
   float m=modulation.data[base+target].y;
   weight+=m*eligibility*(m>=0.0?1.0-weight:weight);
   if(is_input){weight=max(weight,p.input_floor);e.padding=0.0;}
   else{
     e.padding*=1.0-abs(m)*eligibility;
     float drop=decay;
     if(target_role==2)drop*=target_state.z;
     weight=subtract_decay(weight,drop,e.padding);
   }
   e.weight=weight;e.trace=eligibility;
   if(!is_input && weight<=0.0){
      e.target=-1;
      atomicAdd(incoming_degrees.data[base+target],-1);
      // Keep target-side atomics: several sources may die into one target.
      atomicOr(dirty.data[base+target],1);
      deaths++;
   }
   edges.data[index]=e;
 }

 // Each tile decrements the source live degree by its own death count.
 // Atomics are necessary because multiple tiles share the same source.
 if(deaths>0){
   atomicAdd(live_degrees.data[base+n],-deaths);
   atomicAdd(changes.data[6u*agent+0u],deaths);
 }
}
