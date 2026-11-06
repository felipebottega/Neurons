# Neurons

**Neurons** is an experimental artificial-brain simulator built with **Godot 4**. Its research question is whether compact genomes can generate neural systems capable of lifetime adaptation and increasingly general behavior through evolution, without backpropagation or a task-specific learning algorithm.

A genome generates a brain; the brain controls an agent; the agent is evaluated in an environment; and evolution selects genomes for the next generation. **Evolution modifies genomes, while plasticity modifies individual brains during their lifetimes.**

## Theory

### Genome

Let $\mathcal D=\{0,1,2,3\}$. A genome is an ordered tuple of genes:

$$
X=(x^{(1)},\ldots,x^{(n)}),\qquad x^{(r)}\in\mathcal D^{p_r}.
$$

In the current full model, $n=16$: seven genes have 8 digits and nine have 64 digits. Thus

$$
\sum_{r=1}^{16}p_r=7(8)+9(64)=632,\qquad
|\mathcal G|=4^{632}.
$$

A gene is **not** a list of neurons or connections. It is mapped to scalar parameters or spatial fields. Two basic mappings are

$$
F_\beta(x)=\sum_{t=0}^{p-1}\sqrt{1+t}\,\beta^{x_t},
\qquad
G_{\delta_1,\delta_2}(x)=\sum_{t=0}^{p-1}\sin(1+t)\sin(1+x_t)^{\delta_1}\cos(1+x_t)^{\delta_2}.
$$

Spatial functions derived from the genes determine neuronal properties and connectivity. Genetic variation consists of crossover and digit mutations in $\mathcal D$.

### Brain and connectivity

The brain occupies the lattice

$$
\mathcal N=\{0,\ldots,I-1\}\times\{0,\ldots,J-1\}\times\{0,\ldots,K-1\},
\qquad N=IJK.
$$

Directed connections $a\to b$ have weights $w_a^b\in[0,1]$. Genetic functions determine candidate connections and their initial weights, subject to the outgoing-degree limit $N_{\max}$.

Input and output roles are assigned **after provisional connectivity**. With $m=N_{\mathrm{in}}+N_{\mathrm{out}}>0$, define

$$
q = \left\lfloor\frac{N}{m}\right\rfloor, \qquad g=\max\{p \lt q: p \text{ is prime}\}.
$$

The preferred zero-based indices are $rg-1$, for $r=1,\ldots,m$. A circular search selects eligible, unassigned neurons using provisional connectivity. Invalid role-to-role connections are removed, followed by connectivity rescue. Inputs have no incoming neural connections and connect only to internal neurons; outputs receive only from internal neurons and have no outgoing neural connections.

### Neural dynamics

For $s\geq0$, neural signals are normalized by

$$
\sigma(s)=1-e^{-s/\kappa},\qquad \kappa\geq1.
$$

Each non-input neuron has a state $s_t$, genetically determined activation properties, a refractory mechanism, and a recent-activity trace $R_t\in[0,1]$. An active internal neuron transmits a signal through its outgoing connections, scaled by the weight and source polarity $\alpha\in\{-1,+1\}$. Input neurons instead receive external scalar signals and have special transmission rules.

The full model changes connections through four complementary processes:

- **Hebbian reinforcement:** when a connection transmits, $w\leftarrow w+\gamma(1-w)$, with $0\leq\gamma\leq1/2$.
- **Endogenous modulation:** an eligibility trace $e_t$ records recent synaptic use; a spatial modulatory field produces $m_t$. The weight update is $w\leftarrow w+m_te_t(1-w)$ for $m_t\geq0$, or $w\leftarrow w+m_te_tw$ for $m_t<0$.
- **Synaptic decay:** non-input connections can weaken until removed; input-origin connections are protected from ordinary decay.
- **Structural plasticity:** connections may be removed, created, or replaced, subject to connectivity constraints.

**Homeostasis.** With $L=0.05$, $H=0.95$, and $s_H=-\kappa\ln(1-H)$, the high-activity correction applies when $R_t>H$ and $s>s_H$. Define

$$
\lambda_t=\frac{R_t-H}{1-H},\qquad
s\leftarrow s-\lambda_t(s-s_H).
$$

At the same event, every **incoming connection whose source is not an input neuron** undergoes

$$
\boxed{w_a^b\leftarrow(1-\lambda_t(b))w_a^b.}
$$

This synaptic downscaling is independent of excitatory/inhibitory polarity. Low-activity homeostasis corrects the neuronal state but does **not** increase synaptic weights. The input-origin exclusion preserves the special sensory connectivity rules.

### Evolution

For each generation:

$$
\text{genomes}\ \longrightarrow\ \text{brains}\ \longrightarrow\
\text{agent behavior}\ \longrightarrow\ \text{fitness}\ \longrightarrow\
\text{selection, crossover, mutation}.
$$

Selected elite genomes can survive unchanged. Evaluation is task-dependent; the neural equations do not receive an explicit reward or gradient from fitness.

## Implementations

| Version | Brain execution | Scope |
|---|---|---|
| **CPU** | GDScript on CPU | Full model without a native extension. |
| **CPP** | Native C++ via GDExtension | Full model optimized for CPU execution; also contains a GDScript path. |
| **GPU** | GLSL compute shaders | Full model with parallel brain construction and dynamics. |
| **GPU-Model-Settings** | GLSL compute shaders | Configurable GPU variant for controlled experiments and mechanism ablations. |
| **GPU-Vsimple** | GLSL compute shaders | Simplified dynamics with fixed synaptic weights. |

**CPU.** Direct GDScript implementation of brain construction and neuronal dynamics. Useful as an accessible CPU reference.

**CPP.** Native CPU implementation of the full model, integrated with Godot through GDExtension. Requires compiling the extension for the target platform.

**GPU.** Full model implemented with compute shaders and GPU-oriented connectivity storage and update passes.

**GPU-Model-Settings.** Extends GPU with pre-run controls for selected model parameters and mechanisms, including Hebbian plasticity, structural plasticity, synaptic decay, and homeostasis. The homeostasis control governs both neuronal state correction and synaptic downscaling.

**GPU-Vsimple.** Uses genetically generated connectivity, prime-based input/output selection, and signal propagation, but has **fixed weights** and omits the full model's Hebbian plasticity, endogenous modulation, synaptic decay, homeostasis, and structural plasticity. It is a different mathematical model, not a full-model acceleration mode.

## Simulation 6 — Obstacle Proximity Sensors

Simulation 6 extends Simulation 4 while retaining its environment, fitness, and evolutionary curriculum. Its brain has **13 sensory inputs** and **4 motor outputs**:

- **4 target-distance inputs:** cardinal directions, as in Simulation 4.
- **1 binary collision input:** indicates physical contact.
- **8 obstacle-proximity inputs:** equally spaced directions, including diagonals, covering 360°.

Each proximity sensor detects the nearest obstacle surface **or world boundary** along its direction, up to a range of **600 pixels**. With distance $d$ to contact (accounting for the agent's size), its normalized signal is

$$
s(d)=1-\min\left(1,\frac{d}{600}\right).
$$

Thus $s=0$ when no surface is within range and $s=1$ at contact. Simulation 4 remains available unchanged for comparison.

## Simulation 7 — Maze with Extended Obstacle Sensing

Simulation 7 uses Simulation 5's maze, movement, fitness, and difficulty progression, with **independent difficulty state**. Its brain has **12 sensory inputs** and **4 motor outputs**:

- **Inputs 1–4:** target displacement (left, right, up, down), unchanged from Simulation 5.
- **Inputs 5–12:** obstacle proximity (W, E, N, S, NW, NE, SW, SE), with a **600-pixel** range, detecting maze walls and arena boundaries.

The eight proximity inputs **replace**, rather than supplement, Simulation 5's four 220-pixel wall sensors. They use the same normalization $s(d)$ defined for Simulation 6, measuring distance to the first collision surface with the agent's size accounted for. Simulation 5 remains unchanged for comparison.

## Analysis Tools

Depending on the implementation, the project provides 3D brain visualization, neuron/connection inspection, parameter heatmaps, evolutionary fitness and diversity statistics, connectivity statistics, and a **Genetic × Fitness Landscape** view relating brain phenotypes to fitness.

## Performance

Performance depends on brain dimensions, population size, connectivity, enabled mechanisms, and hardware. CPU, CPP, and GPU are distinct computational implementations; no fixed performance ratio is assumed. The long-term objective includes much larger neural populations, subject to memory and throughput constraints.

## Current Status

Neurons is an **active research prototype**, not a production neural-network framework. Current work focuses on mathematical and numerical validation, evolutionary experiments, generalization, performance, and emergent network organization.

## Requirements

- **Godot 4.7.x**.
- Compatible GPU and driver for GPU variants.
- Supported C++ compiler and GDExtension dependencies for the native **CPP** path.
- Sufficient CPU/GPU memory for the selected simulations.

Development has primarily taken place on Windows.

## Running the Project

1. Clone or download the desired implementation.
2. For **CPP**, compile the native GDExtension if using its native execution path.
3. Open the corresponding Godot project and run its simulation scene.
4. Configure brain dimensions, input/output counts, population, mutation rate, and other exposed parameters.
5. For **GPU-Model-Settings**, select the desired model settings before running.

Controls and supported mechanisms differ between variants.

## Research Direction

> Can a compact genome generate a neural system that develops useful internal dynamics, adapts during its lifetime, and evolves toward increasingly general behavior?

The emphasis is on local rules, continuous dynamics, evolving connectivity, internal state, and minimal task-specific learning logic. Success on a fixed fitness task alone does not establish general learning ability.

## Disclaimer

This project is experimental research software. Biological inspiration does **not** imply biological realism. The model deliberately simplifies nervous systems to study artificial adaptive dynamics at manageable computational cost.

## Tips

To compile the **CPP** GDExtension on Windows, from the Godot project root, in the configured build environment:

```bash
conda activate myenv
scons platform=windows target=template_release api_version=4.7
```

The previously used `template_debug` command is discontinued in this project; use a supported build configuration for the installed Godot/GDExtension version.
