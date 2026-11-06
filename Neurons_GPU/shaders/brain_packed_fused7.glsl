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

layout(set=0,binding=1,std430) readonly buffer OffsetsBuffer {uint data[];} offsets;
layout(set=0,binding=2,std430) readonly buffer LengthsBuffer {uint data[];} lengths;
layout(set=0,binding=3,std430) readonly buffer IndicesBuffer {int data[];} indices;
layout(set=0,binding=4,std430) buffer EdgesBuffer {Edge data[];} edges;
// Packed source values avoid two scattered vec4 state loads per synapse.
layout(set=0,binding=5,std430) readonly buffer SourceRuntimeBuffer {vec4 data[];} source_runtime;
layout(set=0,binding=6,std430) writeonly buffer IncomingBuffer {float data[];} incoming;
layout(set=0,binding=7,std430) readonly buffer StatesBuffer {vec4 data[];} states;
layout(set=0,binding=8,std430) readonly buffer ModulationBuffer {vec2 data[];} modulation;
layout(set=0,binding=9,std430) buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=10,std430) buffer Live_degreesBuffer {int data[];} live_degrees;
layout(set=0,binding=11,std430) buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=12,std430) buffer DirtyBuffer {int data[];} dirty;
layout(set=0,binding=13,std430) readonly buffer EdgeConstantsBuffer {vec4 data[];} edge_constants;
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

// The CSR row belongs to one receiver. Every existing live connection occurs
// in exactly one row; consequently edge writes have exactly one owner.
// This kernel replaces both V3.1's spmv and its slot-tiled edges pass.
// The source emission and all neuron states are immutable during the pass.
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint target=gl_GlobalInvocationID.x;
 if(target>=p.total)return;
 uint base=p.neuron_base, eb=p.edge_base;
 uint first=offsets.data[base+target];
 uint count=lengths.data[base+target];
 if(count==0u){incoming.data[base+target]=0.0;return;}
 float sum=0.0;
 int deaths=0;
 // Receiver data is invariant for every incoming synapse in this row.
 vec4 target_state=states.data[3u*(base+target)];
 int receiver_role=int(edge_constants.data[base+target].y);
 float m=modulation.data[base+target].y;
 for(uint k=0u;k<count;k++){
    uint slot=uint(indices.data[eb+first+k]);
    uint source=slot%p.total;
    uint index=eb+slot;
    Edge e=edges.data[index];
    // IMPORTANT: propagate using pre-plasticity weight, including a synapse
    // that will die later in this very iteration.
    vec4 src=source_runtime.data[base+source];
    sum+=src.x*e.weight;

    vec4 source_constants=edge_constants.data[base+source];
    bool is_input=source_constants.y==1.0;
    bool fired=src.z>0.5;
    float eligibility=p.mu*e.trace;
    float weight=e.weight;
    int target_role=is_input?0:receiver_role;
    if(fired){
        float hebbian=source_constants.z;
        weight+=hebbian*(1.0-weight);
        if(!is_input)e.padding*=1.0-hebbian;
        eligibility+=(1.0-p.mu)*src.y*target_state.w;
    }
    weight+=m*eligibility*(m>=0.0?1.0-weight:weight);
    if(is_input){weight=max(weight,p.input_floor);e.padding=0.0;}
    else{
        e.padding*=1.0-abs(m)*eligibility;
        float drop=source_constants.x;
        if(target_role==2)drop*=target_state.z;
        weight=subtract_decay(weight,drop,e.padding);
    }
    e.weight=weight;
    e.trace=eligibility;
    if(!is_input && weight<=0.0){
        e.target=-1;
        // Other receivers can lose edges from this same source.
        atomicAdd(live_degrees.data[base+source],-1);
        deaths++;
    }
    edges.data[index]=e;
 }
 incoming.data[base+target]=sum;
 if(deaths>0){
    // Exactly one invocation owns this receiver; the former per-edge atomics
    // on incoming_degrees and dirty are no longer needed.
    int old_degree=incoming_degrees.data[base+target];
    incoming_degrees.data[base+target]=old_degree-deaths;
    atomicOr(dirty.data[base+target],1);
    atomicAdd(changes.data[6u*agent+0u],deaths);
 }
}
