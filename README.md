# Neurons

**Neurons** is an experimental artificial brain simulator built with **Godot 4**.

The project explores whether complex adaptive behavior can emerge from compact genetic descriptions of neural structure and dynamics. Instead of training a fixed neural network with backpropagation, each brain is generated from a genome, simulated directly, evaluated in an environment, and evolved across generations.

The long-term goal is to investigate systems that can **learn how to learn**, using biologically inspired mechanisms such as local plasticity, neuromodulation, refractory behavior, eligibility traces, and structural plasticity.

## Overview

A genome defines the structure and parameters of a three-dimensional artificial brain.

From that genome, Neurons generates:

- neuron properties
- input and output neuron placement
- local connectivity
- initial synaptic weights
- neural dynamics
- plasticity parameters
- modulatory dynamics
- structural plasticity behavior

The generated brain is simulated as a dynamical system and evaluated through behavioral tasks. Better-performing genomes are selected to produce the next generation.

## Main Ideas

### Genetically generated brains

The complete brain is not stored directly in the genome.

Instead, the genome contains compact numerical genes interpreted as continuous spatial functions over a 3D neural volume. Small genetic changes are intended to produce correspondingly small changes in the generated brain.

This allows evolution to search over brain architectures without encoding every neuron or connection individually.

### Three-dimensional neural space

Neurons occupy positions in a regular 3D mesh.

Each neuron can have genetically determined properties such as:

- activation threshold
- decay factor
- retention factor
- polarity
- Hebbian plasticity rate
- modulatory release
- modulatory sensitivity

Input and output neurons are selected from the generated neural field, while the remaining neurons form the internal network.

### Genetically controlled connectivity

Connections are generated from a spatial compatibility function between source and target neurons.

A source neuron searches a local neighborhood of nearby depth slices and selects candidate targets according to the genetically defined connection function.

The genome therefore controls connectivity indirectly rather than explicitly listing edges, allowing relatively compact genomes to generate large structured networks.

## Neural Dynamics

The simulator includes several biologically inspired mechanisms.

### Signal propagation

Neuron activity is propagated through weighted directed connections.

Signals are normalized before firing decisions and transmission.

### Refractory behavior

After firing, neurons temporarily become harder to activate. This refractory state gradually decays over time.

### Synaptic decay

Connection weights can decay continuously during simulation.

### Hebbian plasticity

Connection weights can change according to local activity.

### Eligibility traces

Connections maintain eligibility traces that allow delayed modulatory signals to affect previous activity.

### Modulatory field

Neurons can release a local modulatory signal into the surrounding neural space. This field spreads through neighboring positions and influences plasticity according to each neuron's modulatory sensitivity.

### Structural plasticity

Connections can be destroyed and replaced during the lifetime of the brain. The network topology can therefore change during simulation instead of remaining fixed after generation.

## Inputs and Outputs

Input neurons receive external scalar signals from the environment.

The external system is responsible for mapping raw observations into values suitable for the neural inputs.

Possible inputs include:

- distance measurements
- movement information
- sensory channels
- internal state variables
- energy
- hunger
- arousal
- other artificial physiological signals

Output neurons expose neural signals that can control movement, actions in the environment, or internal state.

Input neurons connect only to internal neurons, and output neurons receive connections only from internal neurons.

## Evolution

Neurons includes an evolutionary simulation system.

Each generation contains a population of independently generated brains.

The general cycle is:

1. Generate brains from genomes.
2. Simulate each agent.
3. Measure fitness.
4. Select the best-performing genomes.
5. Generate offspring through crossover and mutation.
6. Build a new population.
7. Repeat.

Selected elite genomes survive unchanged, while the remaining population is generated through crossover and mutation.

Only genomes and compact historical statistics need to be preserved between generations. Full brains can always be reconstructed from their genomes.

## Analysis Tools

The project includes tools for inspecting individual brains and the evolutionary process.

### Brain visualization

Brains can be displayed as a 3D neural graph.

Available heatmaps include parameters such as:

- activation threshold
- decay factor
- retention factor
- Hebbian plasticity
- polarity
- neuron type
- incoming connectivity
- connection energy

Individual neurons can also be selected to inspect incoming and outgoing connections.

### Population statistics

The simulator tracks metrics such as:

- mean fitness
- best fitness
- genetic diversity
- connection statistics
- isolated neurons
- rescue failures
- generation time

### Genetic × Fitness Landscape

The simulator can visualize the relationship between selected brain phenotypes and fitness.

Multiple phenotype dimensions are recorded for each generated brain, including directly generated parameters and aggregate network statistics. Two dimensions can be selected to inspect their relationship with fitness as a heatmap.

## Connectivity Rescue

A generated brain may contain isolated neurons.

After initial connectivity is created, a rescue stage attempts to satisfy minimum connectivity requirements.

The current rules require:

- every internal neuron to have at least one incoming connection
- every output neuron to have at least one incoming connection
- every input neuron to have at least one outgoing connection

Rescue connections must still satisfy the structural and genetic connectivity rules.

If these conditions cannot be satisfied, the generated brain is considered invalid.

## Performance

The project currently uses a hybrid implementation:

- **Godot / GDScript** for simulation management, UI, experiments, visualization, and evolutionary control
- **C++ GDExtension** for performance-critical brain generation and neural dynamics

Brain construction and dynamics use native data structures and multithreaded CPU code.

GPU compute is a future optimization target, but the project intentionally validates the complete mathematical model on the CPU first.

The long-term target is the simulation of very large neural systems, potentially reaching millions of neurons.

## Current Status

Neurons is an active experimental project.

Current areas of work include:

- validating the mathematical model
- improving evolutionary experiments
- evaluating generalization rather than memorization
- improving CPU performance
- analyzing emergent neural organization
- improving fitness landscape visualization
- preparing the architecture for future GPU compute execution

The project should currently be considered a research prototype rather than a production neural-network framework.

## Requirements

- **Godot 4.7.x**
- A C++ compiler supported by Godot GDExtension
- A desktop system capable of running large CPU simulations

The project has been developed primarily on Windows.

## Running the Project

1. Clone the repository.
2. Build the native GDExtension used by the project.
3. Open the project in Godot.
4. Run the desired simulation scene.
5. Configure the brain size, population parameters, mutation rate, inputs, outputs, and simulation settings from the interface.

Performance depends strongly on brain size, connectivity, and the selected experiment.

## Research Direction

Neurons is not intended to reproduce conventional deep-learning architectures.

The central question is:

> Can a compact genome generate a neural system that develops useful internal dynamics, adapts during its lifetime, and evolves toward increasingly general behavior?

The project emphasizes:

- local rules
- continuous dynamics
- evolving structure
- internal state
- biologically inspired mechanisms
- minimal task-specific logic

The objective is not simply to optimize behavior on a fixed training set, but to explore neural systems capable of developing more general adaptive strategies.

## Disclaimer

This project is experimental research software.

Its biological inspiration should not be interpreted as a claim of biological realism. The model deliberately simplifies many aspects of real nervous systems in order to study artificial adaptive dynamics in a computationally manageable form.


## Tips
Command to compile GDExtension (at the root level of the Godot project): 
 - conda activate crypto
 - scons platform=windows target=template_debug api_version=4.7