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
layout(set=0,binding=3,std430)  buffer EdgesBuffer {Edge data[];} edges;
layout(set=0,binding=4,std430)  buffer CountsBuffer {int data[];} counts;
layout(set=0,binding=5,std430)  buffer Incoming_degreesBuffer {int data[];} incoming_degrees;
layout(set=0,binding=6,std430)  buffer Live_degreesBuffer {int data[];} live_degrees;
layout(set=0,binding=7,std430)  buffer HeadsBuffer {int data[];} heads;
layout(set=0,binding=8,std430)  buffer LinksBuffer {int data[];} links;
layout(set=0,binding=9,std430)  buffer ChangesBuffer {int data[];} changes;
layout(set=0,binding=10,std430)  buffer Free_headsBuffer {int data[];} free_heads;
layout(set=0,binding=11,std430) readonly buffer Free_nextBuffer {int data[];} free_next;
layout(set=0,binding=12,std430)  buffer Free_countsBuffer {int data[];} free_counts;
int reserve_slot(uint agent,uint source,Params p){
 uint base=p.neuron_base,edgebase=p.edge_base;
 // Free-list nodes are immutable while homeostasis and structural run.
 // The only producer is the earlier prune pass; this prevents ABA during pop.
 int head=atomicAdd(free_heads.data[base+source],0);
 while(head>=0){
  int successor=free_next.data[edgebase+uint(head)*p.total+source];
  int previous=atomicCompSwap(free_heads.data[base+source],head,successor);
  if(previous==head){
   atomicAdd(free_counts.data[base+source],-1);
   return head;
  }
  head=previous;
 }
 int observed=atomicAdd(counts.data[base+source],0);
 while(observed<int(p.stride)){
  int previous=atomicCompSwap(counts.data[base+source],observed,observed+1);
  if(previous==observed){
   return observed;
  }
  observed=previous;
 }
 return -1;
}
void main(){
 uint agent=gl_GlobalInvocationID.y;
 Params p=parameters.data[agent];
 uint n=gl_GlobalInvocationID.x; uint base=p.neuron_base; uint edgebase=p.edge_base;

 if(n>=p.total)return;
 if(roles.data[base+n]!=0||states.data[3u*(base+n)+2u].w<0.5)return;
 // Structural plasticity samples the whole brain, excluding its source.
 // The skipped-index mapping needs one random draw and cannot select self.
 if(p.total<=1u)return;
 uint options=p.total-1u;
 uint slot=neural_random(n, p.iteration, 4271u, agent)%options;
 uint target=slot>=n?slot+1u:slot;
 if(target>=p.total||roles.data[base+target]==1)return;
 // All three criteria refer to iteration t: prepare stores sigma(s_t)
 // in a.w and R_t in a.z; state advance preserves both fields.
 const float PROXIMITY_EPSILON = 0.01;
 float source_trace=states.data[3u*(base+n)].z;
 float target_trace=states.data[3u*(base+target)].z;
 if(source_trace+target_trace<=p.structural)return;
 if(abs(source_trace-target_trace)>=PROXIMITY_EPSILON)return;
 float source_signal=states.data[3u*(base+n)].w;
 float target_signal=states.data[3u*(base+target)].w;
 if(abs(source_signal-target_signal)>=PROXIMITY_EPSILON)return;
 uint start=edgebase+n;int count=counts.data[base+n];
 for(int k=0;k<count;k++)if(edges.data[start+uint(k)*p.total].target==int(target))return;
 int pos=reserve_slot(agent,n,p);
 // No structural change is made at N_max. This is a normal saturation,
 // recorded separately from structural corruption in changes[3].
 if(pos<0){atomicAdd(changes.data[6u*agent+4u],1);return;}
 uint local_index=uint(pos)*p.total+n,at=edgebase+local_index;
 edges.data[at]=Edge(int(target),0.25*p.structural,0.0,0.0);
 links.data[at]=atomicExchange(heads.data[base+target],int(local_index));
 atomicAdd(incoming_degrees.data[base+target],1);atomicAdd(live_degrees.data[base+n],1);
 atomicAdd(changes.data[6u*agent+2u],1);

}
