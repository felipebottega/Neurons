# Neurons

A small experimental simulation of **artificial neuronal evolution** built with **Godot 4** and a **Python WebSocket server**.

The project explores a different path from standard neural networks: instead of training a fixed architecture, each agent owns a **genetically encoded 3D brain**. That brain is then used during a simulation where agents move, sense the environment, consume food, reproduce, and pass on modified neural structures to offspring.

<p align="center">
   <img width="1100" src="https://github.com/user-attachments/assets/ed45fd1d-0188-4cc0-899f-7561a738f689" />
</p>

## Overview

The system is organized around three layers:

1. **Genetics**  
   Each agent has a genome that defines its brain structure and behavior.
2. **Brain**  
   The genome is decoded into a 3D network of neurons, inputs, outputs, thresholds, regions, and connections.
3. **Evolutionary simulation**  
   Agents interact with the environment, survive, reproduce, and generate offspring with inherited and mutated traits.

## Main idea

Each blob is a physical agent in the Godot scene. It has:

- a position and movement in 2D space,
- energy that decreases over time,
- input signals from the environment,
- output actions such as moving and rotating,
- genetically defined neural connectivity.

The Python server processes the neural logic, while Godot handles the world simulation, physics, collision detection, rendering, and user interface.

## Features

- Genetic generation of neural brains
- 3D brain topology encoded by a genome
- Directional neuron-to-neuron connections
- Input and output neurons defined by genetics
- Dynamic connection strength updates over time
- Energy-based survival and reproduction
- Food-based environmental interaction
- Offspring generation with inheritance and mutation
- WebSocket communication between Godot and Python
- History logging for neural activity and connection growth
- Plot scripts for visualizing simulation data

## Project structure

- `server.py` — Python FastAPI WebSocket server responsible for brain generation, processing, and offspring creation
- `scripts/` — Godot scripts for blobs, food, simulation flow, and networking
- `scenes/` — Godot scenes for the main world, blobs, food, HUD, and helper nodes
- `autoload/` — Global state shared across the simulation
- `plot_neurons.py` — Visualization script for neuron activity and connection history
- `artigo/` — LaTeX article describing the model conceptually

## How it works

### 1. Brain generation

When a blob is created, the server generates a new brain genome.  
The genome defines:

- brain dimensions,
- input neuron mapping,
- output neuron mapping,
- output thresholds,
- neuron regions,
- initial connection graph,
- connection probability,
- neuron threshold,
- decay and timing parameters.

### 2. Simulation loop

Each simulation step:

- Godot gathers the current inputs from each blob,
- sends them to the Python server through the WebSocket,
- the server evaluates neural propagation,
- outputs are returned,
- Godot applies those outputs to movement and behavior.

### 3. Evolution

When two blobs collide under the right conditions, they can create offspring.  
The offspring inherits parts of both parents’ genomes and may also receive random mutations.

## Inputs

Each blob receives a fixed set of inputs, including:

- linear velocity components,
- angular velocity,
- rotation,
- energy,
- inverse energy,
- food proximity,
- collision state,
- inverse distance to food,
- previous outputs.

## Outputs

The current output set controls the blob’s actions:

- move right
- move left
- rotate clockwise
- rotate counterclockwise

## Visualization

The repository also includes scripts to inspect the simulation data:

- `blob_history.txt` stores neural activity history
- `num_connections_history.txt` stores connection counts over time
- `plot_neurons.py` generates HTML visualizations from these logs

## Requirements

- **Godot 4.x**
- **Python 3.x**
- Python packages used by the server and plotting scripts, such as:
  - `fastapi`
  - `uvicorn`
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `plotly`

## Running the project

### 1. Start the Python server

```bash
python server.py
```

### 2. Run the Godot project

<p align="center">
   <img width="1100" src="https://github.com/user-attachments/assets/7f5f1077-d107-4f8a-b059-aeeaecd5bceb" />
</p>

## Future workd
- [ ] Implement parallelism.
- [ ] Weak environmental observability: lack of explicit food direction/gradient.
- [ ] Eliminate blob_0.
- [ ] Improve the reward, as it is not used as a direct neural learning signal.
