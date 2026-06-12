# The Hunters and the Hive - Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Running the Simulation](#running-the-simulation)
4. [Configuration Options](#configuration-options)
5. [Running Experiments](#running-experiments)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

"The Hunters and the Hive" is a swarm robotics simulation that models pursuit-evasion scenarios using swarm intelligence algorithms:

- **Pursuers (Hunters)**: Always use Particle Swarm Optimization (PSO) to chase evaders
- **Evaders (The Hive)**: Can use either:
  - **Ant Colony Optimization (ACO)**: Pheromone-based evasion with wide-area search
  - **PSO**: Particle swarm maximizing distance from pursuers

The simulation demonstrates how different algorithms handle:
- Swarm coordination
- Obstacle avoidance
- Dynamic target tracking

---

## Installation

### Step 1: Check Python Version
```bash
python --version
```
Requires Python 3.8 or higher.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computing
- `matplotlib` - Visualization and animation
- `scipy` - Statistical analysis

---

## Running the Simulation

### Basic Run (Interactive Visualization)
```bash
python swarm_pursuit_evasion_3.py
```

This launches an animated window showing:
- **Red circles**: Pursuers
- **Blue squares**: Active evaders
- **Gray X marks**: Captured evaders
- **Gray boxes**: Obstacles
- **Red tint**: ACO pheromone trails

### Programmatic Usage

```python
from swarm_pursuit_evasion_3 import (
    SimulationConfig,
    SwarmSimulation,
    create_scenario
)

# Create configuration
config = create_scenario("open_field")
config.evader_type = "ACO"  # or "PSO"
config.n_pursuers = 5
config.n_evaders = 3

# Run simulation
sim = SwarmSimulation(config)
metrics = sim.run(visualize=True)  # Set False for headless

print(f"Capture rate: {metrics['capture_rate']}")
print(f"Avg survival: {metrics['avg_survival_time']}")
```

---

## Configuration Options

### Environment Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `world_size` | tuple | (100, 100) | (width, height) |
| `dt` | float | 0.5 | Time step size |
| `max_steps` | int | 500 | Maximum simulation steps |
| `obstacles` | list | [] | List of (x, y, width, height) |

### Pursuer (PSO) Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_pursuers` | 5 | Number of pursuers |
| `pursuer_max_speed` | 2.0 | Maximum velocity |
| `pursuer_sensing_radius` | 30.0 | Communication range |
| `pursuer_capture_radius` | 2.0 | Capture distance |
| `pursuer_inertia` | 0.7 | ω (omega) weight |
| `pursuer_cognitive` | 1.5 | c1 (cognitive) weight |
| `pursuer_social` | 1.5 | c2 (social) weight |

### Evader Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_evaders` | 3 | Number of evaders |
| `evader_type` | "ACO" | "ACO" or "PSO" |
| `evader_max_speed` | 1.8 | Maximum velocity |
| `evader_sensing_radius` | 25.0 | Detection range |

### ACO-Specific Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `aco_grid_resolution` | 10 | Pheromone grid cell size |
| `aco_pheromone_decay` | 0.95 | Decay per step |
| `aco_pheromone_deposit` | 10.0 | Pheromone amount |
| `aco_danger_weight` | 2.0 | Danger weighting |

### Pre-built Scenarios

```python
config = create_scenario("open_field")      # No obstacles
config = create_scenario("sparse_obstacles") # 4 obstacles
config = create_scenario("dense_maze")       # 11 obstacles
```

---

## Running Experiments

### Command Line Interface

```bash
# Run all experiments (1, 2, 3)
python planned_experiments.py

# Run specific experiment(s)
python planned_experiments.py --experiments 1
python planned_experiments.py --experiments 2 3

# Adjust number of runs
python planned_experiments.py --experiments 1 --n-runs 10

# Custom snapshot steps
python planned_experiments.py --snapshot-step 50 100 200
```

### Experiment Descriptions

#### Experiment 1: Parameter Sensitivity
Analyzes how swarm size, inertia weight, and sensing radius affect performance.

```bash
python planned_experiments.py --experiments 1 --n-runs 30
```

#### Experiment 2: Environmental Complexity
Compares algorithm performance across different obstacle configurations.

```bash
python planned_experiments.py --experiments 2 --n-runs 30
```

#### Experiment 3: Scalability
Tests pursuer-to-evader ratios: 1:1, 2:1, 1:2.

```bash
python planned_experiments.py --experiments 3 --n-runs 30
```

### Output Files

Experiments generate:
- `experiment_results/experiment1_parameter_sensitivity.json`
- `experiment_results/experiment2_environmental_complexity.json`
- `experiment_results/experiment3_scalability.json`
- `experiment_results/snapshots/` - Visualizations at steps 30, 100, 250, 450

---

## Understanding Results

### Metrics

| Metric | Description |
|--------|-------------|
| `capture_rate` | Fraction of evaders captured (0-1) |
| `n_captured` | Number of evaders captured |
| `avg_survival_time` | Mean steps before capture |
| `total_steps` | Total simulation steps |
| `capture_times` | Individual capture timestamps |

### Statistical Tests

Experiments include Mann-Whitney U tests comparing:
- Different parameter values within same algorithm
- ACO vs PSO in same scenario
- Different scenarios

Results are significant if p < 0.05.

---

## Troubleshooting

### Window Doesn't Appear
- Ensure matplotlib backend supports GUI (try `%matplotlib qt`)
- Run with `visualize=False` for headless mode

### Slow Performance
- Reduce `max_steps`
- Use fewer agents
- Close other graphical applications

### All Evaders Escape
- Increase pursuer speed: `config.pursuer_max_speed = 2.5`
- Increase number of pursuers: `config.n_pursuers = 10`
- Increase pursuer inertia for faster convergence

### All Evaders Captured Quickly
- Increase evader speed: `config.evader_max_speed = 2.2`
- Add obstacles: use `dense_maze` scenario
- Use ACO evaders for pheromone-based evasion

### Missing Dependencies
```bash
pip install --upgrade numpy matplotlib scipy
```

---

## Example: Custom Scenario

```python
from swarm_pursuit_evasion_3 import SimulationConfig, SwarmSimulation

# Custom configuration
config = SimulationConfig()
config.world_size = (150, 150)
config.n_pursuers = 8
config.n_evaders = 4
config.evader_type = "ACO"
config.pursuer_inertia = 0.6  # More agile pursuers

# Custom obstacles
config.obstacles = [
    (20, 20, 30, 30),
    (100, 80, 20, 40),
    (50, 100, 40, 10)
]

# Run
sim = SwarmSimulation(config)
metrics = sim.run(visualize=True)
```

---

## Contact

For questions or issues, please open an issue on GitHub.