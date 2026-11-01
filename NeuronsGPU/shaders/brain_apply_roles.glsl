#[compute]
#version 450

// Reflect input-role phenotype overrides in the same GPU neuron data that
// the dynamics backend will consume. Role assignment itself is CPU-ranked.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer Neurons {
    vec4 data[];
} neurons;
layout(set = 0, binding = 1, std430) readonly buffer Roles {
    int data[];
} roles;
layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint reserved;
} p;
void main() {
    uint neuron = gl_GlobalInvocationID.x;
    if (neuron >= p.I * p.J * p.K || roles.data[neuron] != 1) return;
    // threshold = 0, retention = 0, polarity = +1 for input neurons.
    vec4 fields = neurons.data[2u * neuron];
    fields.x = 0.0;
    fields.z = 0.0;
    fields.w = 1.0;
    neurons.data[2u * neuron] = fields;
}
