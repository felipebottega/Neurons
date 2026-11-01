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
};
layout(set=0,binding=0,std430) readonly buffer ParameterBlock {Params data[];} parameters;
layout(push_constant,std430) uniform BrainIdentity {uint agent_id;} pc;
uint plane_size(Params p){return p.I*p.J;}
uint nb(Params p,uint agent){return 0u;}
uint eb(Params p,uint agent){return 0u;}

layout(set=0,binding=1,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=2,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=3,std430) readonly buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=4,std430) readonly buffer ModulationBuffer {vec2 data[];} modulation;
layout(set=0,binding=5,std430) buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=6,std430) readonly buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=7,std430) buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=8,std430) buffer Live_degreesBuffer {int data[];} live_degrees;
layout(set=0,binding=9,std430) writeonly buffer TransmissionsBuffer {vec2 data[];} transmissions;
layout(set=0,binding=10,std430) buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=11,std430) buffer Rescue_flagsBuffer {int data[];} rescue_flags;
layout(set=0,binding=12,std430) buffer DirtyBuffer {int data[];} dirty;
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
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x;
 if(n>=p.total)return;

 // Every outgoing edge is owned by exactly this invocation. A zero live
 // degree means all dead edges have already been detached by gather, so no
 // transmission needs to be generated for them.
 int count=counts.data[n];
 if(count<=0 || live_degrees.data[n]<=0)return;

 vec4 phenotype0=phenotype.data[2u*n],phenotype1=phenotype.data[2u*n+1u];
 vec4 a=states.data[3u*n],c=states.data[3u*n+2u];
 bool is_input=roles.data[n]==1;
 bool fired=c.w>0.5;
 float multiplier=phenotype0.w*(is_input?p.input_gain*a.x:a.w);
 float decay=is_input?0.0:exp(-12.0/phenotype0.y);
 float hebbian=phenotype1.x;
 uint start=n*p.stride;
 int deaths=0;

 for(int k=0;k<count;k++){
   uint index=start+uint(k);
   // Avoid fetching the other 12 bytes of a dead edge. Dead slots are not
   // in incoming lists after gather and need no repeated null transmission.
   int target_id=edges.data[index].target;
   if(target_id<0 || uint(target_id)>=p.total)continue;
   uint target=uint(target_id);
   Edge e=edges.data[index];
   float signal=fired?multiplier*e.weight:0.0;
   // This transmission belongs to the current step even if the edge dies
   // below: gather consumes it before detaching the dead edge.
   transmissions.data[index]=vec2(float(target_id),signal);

   float eligibility=p.mu*e.trace;
   float weight=e.weight;
   // The target role and state do not change during the edges stage.
   // Reuse one vec4 when both Hebbian eligibility and output attenuation
   // need the target state, without loading it for silent non-output edges.
   int target_role=is_input?0:roles.data[target];
   bool need_target_state=fired || target_role==2;
   vec4 target_state=vec4(0.0);
   if(need_target_state)target_state=states.data[3u*target];
   if(fired){
     weight+=hebbian*(1.0-weight);
     if(!is_input)e.padding*=1.0-hebbian;
     eligibility+=(1.0-p.mu)*a.w*target_state.w;
   }
   float m=modulation.data[target].y;
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
      int prev=atomicAdd(incoming_degrees.data[target],-1);
      if(prev==1)atomicOr(rescue_flags.data[target],1);
      // Keep target-side atomics: several sources may die into one target.
      atomicOr(dirty.data[target],1);
      deaths++;
   }
   edges.data[index]=e;
 }

 // Only this invocation updates the live degree of source n in this stage.
 // Coalesce its atomics, and the global death counter, once per source.
 if(deaths>0){
   live_degrees.data[n]-=deaths;
   atomicAdd(changes.data[0u],deaths);
 }
}
