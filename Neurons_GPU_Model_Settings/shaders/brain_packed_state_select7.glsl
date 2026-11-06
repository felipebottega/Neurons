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
 uint radius_i, radius_j, radius_k;
 float mean_initial_incoming; uint reserved1;
 float fatigue_divisor; float decay_exponent; uint model_flags; uint reserved2;
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


layout(set=0,binding=1,std430) readonly buffer PhenotypeBuffer {vec4 data[];} phenotype;
layout(set=0,binding=2,std430) readonly buffer RolesBuffer {int data[];} roles;
layout(set=0,binding=3,std430)  buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=4,std430) readonly buffer ModulationBuffer {vec2 data[];} modulation;
layout(set=0,binding=5,std430) readonly buffer IncomingBuffer {float data[];} incoming;
layout(set=0,binding=6,std430) readonly buffer OutputReverseBuffer {int data[];} output_reverse;
layout(set=0,binding=7,std430) writeonly buffer OutputStreamBuffer {float data[];} output_stream;
layout(set=0,binding=8,std430) buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=9,std430) writeonly buffer Rescue_sourcesBuffer {int data[];} rescue_sources;
layout(set=0,binding=10,std430) readonly buffer SourceRuntimeBuffer {vec4 data[];} source_runtime;
layout(set=0,binding=11,std430) readonly buffer OffsetsBuffer {uint data[];} offsets;
layout(set=0,binding=12,std430) readonly buffer LengthsBuffer {uint data[];} lengths;
layout(set=0,binding=13,std430) readonly buffer IndicesBuffer {int data[];} indices;
layout(set=0,binding=14,std430) buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=15,std430) buffer LiveDegreesBuffer {int data[];} live_degrees;
layout(set=0,binding=16,std430) buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=17,std430) buffer DirtyBuffer {int data[];} dirty;
// Postsynaptic homeostatic downscaling. The receiver's CSR row uniquely
// owns each live incoming edge; the ELLPACK slot gives its physical address.
// The CSR may still contain connections killed by the earlier synapse pass,
// so ignore rows whose edge.target has already been invalidated.
// Weight updates precede structural creation and CSR reconstruction.
void scale_incoming_synapses(Params p, uint agent, uint target, float factor) {
    uint base = p.neuron_base;
    uint eb = p.edge_base;
    uint first = offsets.data[base + target];
    uint count = lengths.data[base + target];
    int deaths = 0;
    for (uint k = 0u; k < count; k++) {
        int local_index = indices.data[eb + first + k];
        if (local_index < 0) continue;
        uint slot = uint(local_index);
        uint at = eb + slot;
        Edge edge = edges.data[at];
        if (edge.target != int(target)) continue;
        uint source = slot % p.total;
        // Sensory input synapses are excluded from synaptic homeostasis.
        // Do not modify either the weight or its passive-decay residual.
        if (roles.data[base + source] == 1) continue;
        // Scale the compensated passive-decay residual together with weight.
        // Eligibility traces are unaffected by homeostatic weight scaling.
        edge.weight *= factor;
        edge.padding *= factor;
        if (edge.weight <= 0.0) {
            edge.target = -1;
            edge.weight = 0.0;
            edge.padding = 0.0;
            atomicAdd(live_degrees.data[base + source], -1);
            deaths++;
        }
        edges.data[at] = edge;
    }
    if (deaths > 0) {
        // One invocation owns each destination, so the incoming-degree write
        // is race-free. Multiple destinations may kill edges of one source.
        incoming_degrees.data[base + target] -= deaths;
        atomicOr(dirty.data[base + target], 1);
        atomicAdd(changes.data[6u * agent], deaths);
    }
}

void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base; uint edgebase=p.edge_base;

 if(n>=p.total)return;
 vec4 a=states.data[3u*(base+n)],b=states.data[3u*(base+n)+1u],c=states.data[3u*(base+n)+2u];
 // Previous output shader read the firing gate and normalized signal.
 // State updates a.x, b.w, c.x and c.z, but neither a.w nor c.w.
 // Thus writing here produces exactly the old post-plasticity output value.
 int output_channel=output_reverse.data[base+n];
 if(output_channel>=0){
   output_stream.data[uint(output_channel)]=c.w>0.5?a.w:0.0;
 }
 bool is_input=roles.data[base+n]==1;
 // V10: use the live incoming degree already maintained by synaptic
 // creation/death. This is one O(1) lookup; no scan of incoming edges.
 // The developmental mean remains fixed, and input neurons retain M=0.
 float retention=0.0;
 if(!is_input){
    float genetic=phenotype.data[2u*(base+n)].z;
    int degree=max(incoming_degrees.data[base+n],0);
    if(degree==0)retention=1.0;
    else if(p.mean_initial_incoming>0.0){
       float mean_degree=p.mean_initial_incoming;
       retention=genetic+(1.0-genetic)*(mean_degree/(mean_degree+float(degree)));
    }else retention=genetic; // No initial internal input degree: avoid 0/0.
 }
 float v=max(0.0,retention*a.x+incoming.data[base+n]);
 if(!is_input && (p.model_flags & 2u)!=0u){
    float act=a.z;
    if(act<0.05&&v<p.homeo_low)v+=(0.05-act)/0.05*(p.homeo_low-v);
    else if(act>0.95&&v>p.homeo_high){
       float correction=(act-0.95)/0.05;
       v-=correction*(v-p.homeo_high);
       // Apply only when the high-activity neuronal homeostasis is active.
       // Scale all non-input incoming connections, regardless of source polarity.
       scale_incoming_synapses(p,agent,n,clamp(1.0-correction,0.0,1.0));
    }
 }
 a.x=v;b.w=incoming.data[base+n];c.x=modulation.data[base+n].x;c.z=modulation.data[base+n].y;
 states.data[3u*(base+n)]=a;states.data[3u*(base+n)+1u]=b;states.data[3u*(base+n)+2u]=c;
 // This receiver's state writes preserve a.z. Selection reads the immutable
 // source_runtime.w activity cache instead of racing another thread's writes.
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
   float activity=source_runtime.data[base+src].w;
   if(activity>highest){highest=activity;best=int(src);tied=1u;}
   else if(activity==highest){tied++;if(neural_random(n, p.iteration, src ^ 7919u, agent)%tied==0u)best=int(src);}
 }
 rescue_sources.data[base+n]=best;


}
