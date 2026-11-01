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
layout(push_constant,std430) uniform BrainIdentity {
 uint agent_id;
 uint radius_i, radius_j, radius_k; // V_init from the original brain genome.
} pc;
uint plane_size(Params p){return p.I*p.J;}
uint nb(Params p,uint agent){return 0u;}
uint eb(Params p,uint agent){return 0u;}
// Philox2x32-10: counter-based pseudorandom generator (32-bit arithmetic).
// The key separates brains; the counters separate neurons, iterations and draws.
// No shared RNG state or atomics: reproducible regardless of GPU dispatch order.
uint neural_random(uint neuron, uint iteration, uint draw, uint agent) {
    uint x = neuron;
    uint y = iteration ^ (draw * 0x9E3779B9u);
    uint key = 0xA341316Cu + agent * 0xBB67AE85u;
    for (int round_index = 0; round_index < 10; round_index++) {
        uint hi, lo;
        umulExtended(x, 0xD256D193u, hi, lo);
        uint next_x = hi ^ key ^ y;
        y = lo;
        x = next_x;
        key += 0x9E3779B9u;
    }
    return x;
}
layout(set=0,binding=1,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=2,std430) readonly buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=3,std430) readonly buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=4,std430) readonly buffer Rescue_flagsBuffer {int data[];} rescue_flags;
layout(set=0,binding=5,std430) writeonly buffer Rescue_sourcesBuffer {int data[];} rescue_sources;
layout(set=0,binding=6,std430) writeonly buffer DemandBuffer {int data[];} demand;
layout(set=0,binding=7,std430) buffer RequiredBuffer {int data[];} required;
void main(){
 uint agent=pc.agent_id;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=0u; uint edgebase=0u;

 if(n>=p.total)return;
 // Former capacity stage 0: independent neuron-local clearing and one
 // per-brain required-stride initialization. The following capacity pass
 // runs after a dispatch barrier and can safely accumulate demand.
 demand.data[base+n]=0;
 if(n==0u)required.data[agent]=int(p.stride);
 rescue_sources.data[base+n]=-1;
 if(roles.data[base+n]==1||incoming_degrees.data[base+n]!=0||rescue_flags.data[base+n]==0)return;
 // The initial search region is a rectangular box around the target.
 // The same gene y fixes its radii for every neuron, making the source-
 // target membership symmetric. Restrict candidates to internal neurons.
 uint plane=plane_size(p);
 uint target_i=n%p.I, target_j=(n/p.I)%p.J, target_k=n/plane;
 uint low_i=target_i>pc.radius_i?target_i-pc.radius_i:0u;
 uint low_j=target_j>pc.radius_j?target_j-pc.radius_j:0u;
 uint low_k=target_k>pc.radius_k?target_k-pc.radius_k:0u;
 uint high_i=min(p.I-1u,target_i+pc.radius_i);
 uint high_j=min(p.J-1u,target_j+pc.radius_j);
 uint high_k=min(p.K-1u,target_k+pc.radius_k);
 int best=-1;float highest=-1.0;uint tied=0u;
 for(uint k=low_k;k<=high_k;k++)for(uint j=low_j;j<=high_j;j++)for(uint i=low_i;i<=high_i;i++){
   uint src=i+p.I*(j+p.J*k);
   if(src==n||roles.data[base+src]!=0)continue;
   float activity=states.data[3u*(base+src)].z;
   if(activity>highest){highest=activity;best=int(src);tied=1u;}
   else if(activity==highest){tied++;if(neural_random(n, p.iteration, src ^ 7919u, agent)%tied==0u)best=int(src);}
 }
 rescue_sources.data[base+n]=best;

}
