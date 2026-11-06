#[compute]
#version 450

// Phenotypic scalar fields of the 64-digit genes from the Neurons article.
// All data is indexed spatially: n = i + I*j + I*J*k.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Eight 64-digit genes in this order:
// threshold, decay, retention, polarity, Hebbian, release, sensitivity,
// connections (A for H). Neuron roles are chosen by the prime-gap rule.
layout(set = 0, binding = 0, std430) readonly buffer Genes {
    int digit[];
} genes;

// Two vec4s per neuron (8 values). Unused fields are not allocated.
// 0: threshold, decay, retention, polarity
// 1: Hebbian rate, release factor, sensitivity, connection energy.
layout(set = 0, binding = 1, std430) writeonly buffer Neurons {
    vec4 data[];
} neurons;

layout(push_constant, std430) uniform Params {
    uint I;
    uint J;
    uint K;
    uint total;
} p;

const float PI = 3.14159265358979323846;
// Fixed spatial scale in lattice steps; must match Functions.A_SPATIAL_SCALE.
const float A_SPATIAL_SCALE = 16.0;
// Frequencies omega_r = sqrt(r+1) for r=0,1,2,3 (p=64 => n=4).
// The irrational frequency ratios remove the shared period of integer modes.
const vec4 A_FREQUENCIES = vec4(1.0, 1.4142135623730951,
                               1.7320508075688772, 2.0);

// Compute the three spatial bases once per neuron, rather than repeating
// cosine evaluations for every gene and spatial mode.
vec4 spatial_basis(uint coordinate) {
    return cos((PI * float(coordinate) / A_SPATIAL_SCALE) * A_FREQUENCIES);
}

float scalar_field(int gene_index, vec4 basis_a, vec4 basis_b, vec4 basis_c) {
    float result = 0.0;
    // p=64 and n=4. Keep the mode order t=q+4s+16r exactly.
    for (int r = 0; r < 4; r++) {
        for (int s = 0; s < 4; s++) {
            for (int q = 0; q < 4; q++) {
                int t = q + 4 * s + 16 * r;
                float alpha = (2.0 * float(genes.digit[gene_index * 64 + t]) / 3.0) - 1.0;
                result += alpha * basis_a[r] * basis_b[s] * basis_c[q];
            }
        }
    }
    return tanh(result * 0.125); // 1/sqrt(64)
}

void main() {
    uint n = gl_GlobalInvocationID.x;
    if (n >= p.total) return;

    uint i = n % p.I;
    uint j = (n / p.I) % p.J;
    uint k = n / (p.I * p.J);

    vec4 basis_a = spatial_basis(i);
    vec4 basis_b = spatial_basis(j);
    vec4 basis_c = spatial_basis(k);
    vec4 origin_basis = vec4(1.0); // cos(0) for every spatial mode

    float threshold = 0.5 * (1.0 + scalar_field(0, basis_a, basis_b, basis_c));
    float decay = 1.5 + scalar_field(1, basis_a, basis_b, basis_c);
    float retention = (1.0 - exp(-1.0 - scalar_field(2, basis_a, basis_b, basis_c))) /
                      (1.0 - exp(-2.0));
    float baseline_polarity = scalar_field(3, origin_basis, origin_basis, origin_basis);
    float polarity = scalar_field(3, basis_a, basis_b, basis_c) >= baseline_polarity ? 1.0 : -1.0;
    float hebbian = 0.25 * (1.0 + scalar_field(4, basis_a, basis_b, basis_c));
    float release_factor = 0.5 * (1.0 + scalar_field(5, basis_a, basis_b, basis_c));
    float sensitivity = scalar_field(6, basis_a, basis_b, basis_c);
    float connection_energy = scalar_field(7, basis_a, basis_b, basis_c);

    neurons.data[2u * n] = vec4(threshold, decay, retention, polarity);
    neurons.data[2u * n + 1u] = vec4(hebbian, release_factor, sensitivity, connection_energy);
}
