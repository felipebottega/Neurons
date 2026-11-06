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
// Uniform float in [0,1), using the upper 24 random bits.
float neural_random_float(uint neuron, uint iteration, uint draw, uint agent) {
    return float(neural_random(neuron, iteration, draw, agent) >> 8u) * (1.0 / 16777216.0);
}
layout(set=0,binding=1,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=2,std430) readonly buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=3,std430) readonly buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=4,std430) writeonly buffer Rescue_sourcesBuffer {int data[];} rescue_sources;
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base; uint edgebase=p.edge_base;

 if(n>=p.total)return;
 rescue_sources.data[base+n]=-1;
 // Every disconnected non-input neuron remains eligible on every step.
 // In particular, a missed probability gate does not permanently strand an internal
 // neuron: its incoming degree stays zero until it receives a synapse.
 int role=roles.data[base+n];
 if(role==1 || incoming_degrees.data[base+n]!=0)return;
 // Output neurons always attempt reconnection. Internal neurons get an
 // independent, per-neuron/per-iteration Bernoulli(0.01%) trial. The RNG is
 // counter-based; no global state, contention or CPU readback is required.
 const float INTERNAL_RECONNECT_PROBABILITY = 0.0001; // 0.01%
 if(role==0 && neural_random_float(n,p.iteration,0x5245434Fu,agent)>=INTERNAL_RECONNECT_PROBABILITY)return;
 // The initial search region is a rectangular box around the target.
 // The same gene y fixes its radii for every neuron, making the source-
 // target membership symmetric. Restrict candidates to internal neurons.
 uint plane=plane_size(p);
 uint target_i=n%p.I, target_j=(n/p.I)%p.J, target_k=n/plane;
 uint low_i=target_i>p.radius_i?target_i-p.radius_i:0u;
 uint low_j=target_j>p.radius_j?target_j-p.radius_j:0u;
 uint low_k=target_k>p.radius_k?target_k-p.radius_k:0u;
 uint high_i=min(p.I-1u,target_i+p.radius_i);
 uint high_j=min(p.J-1u,target_j+p.radius_j);
 uint high_k=min(p.K-1u,target_k+p.radius_k);
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
