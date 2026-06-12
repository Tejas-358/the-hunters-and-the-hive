"""
The Hunters and the Hive: Pursuit-Evasion Swarm Intelligence Simulation
Group 7 – Tejas Dalvi & Ravi Moelchand

This module implements competing swarms in a 2D environment:
- PSO-driven pursuers
- ACO or PSO-driven evaders
- Configurable obstacles and environment complexity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from datetime import datetime


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    # Environment
    world_size: Tuple[int, int] = (100, 100)
    dt: float = 0.5  # Time step
    max_steps: int = 500

    # Pursuers (PSO)
    n_pursuers: int = 5
    pursuer_max_speed: float = 2.0
    pursuer_sensing_radius: float = 30.0
    pursuer_capture_radius: float = 2.0
    pursuer_inertia: float = 0.7
    pursuer_cognitive: float = 1.5
    pursuer_social: float = 1.5

    # Evaders
    n_evaders: int = 3
    evader_max_speed: float = 1.8
    evader_sensing_radius: float = 25.0
    evader_type: str = "ACO"  # "ACO" or "PSO"

    # ACO-specific parameters
    aco_grid_resolution: int = 10  # Grid cell size
    aco_pheromone_decay: float = 0.95
    aco_pheromone_deposit: float = 10.0
    aco_danger_weight: float = 2.0

    # PSO evader parameters (if using PSO evaders)
    evader_inertia: float = 0.6
    evader_cognitive: float = 1.2
    evader_social: float = 1.8

    # Obstacles
    obstacles: List[Tuple[float, float, float, float]] = None  # (x, y, width, height)


class Agent:
    """Base class for swarm agents"""

    def __init__(self, position: np.ndarray, max_speed: float, world_size: Tuple[int, int]):
        self.position = position.astype(float)
        self.velocity = np.random.randn(2) * 0.5
        self.max_speed = max_speed
        self.world_size = world_size
        self.captured = False

    def apply_boundary(self):
        """Keep agents within world boundaries"""
        self.position = np.clip(self.position, 0, self.world_size)

    def limit_speed(self):
        """Limit velocity to maximum speed"""
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
    
    def obstacle_collision(self, obstacles):
        for obs in obstacles:
            ox, oy, ow, oh = obs
            if ox < self.position[0] < ox + ow and oy < self.position[1] < oy + oh:
                # Push out toward nearest edge
                edges = {
                    'left':   (self.position[0] - ox,          np.array([ox - 0.1, self.position[1]])),
                    'right':  (ox + ow - self.position[0],     np.array([ox + ow + 0.1, self.position[1]])),
                    'bottom': (self.position[1] - oy,          np.array([self.position[0], oy - 0.1])),
                    'top':    (oy + oh - self.position[1],     np.array([self.position[0], oy + oh + 0.1])),
                }
                nearest = min(edges.values(), key=lambda x: x[0])
                self.position = nearest[1]
                self.velocity *= 0.1  # kill momentum


class Pursuer(Agent):
    """PSO-based pursuer agent"""

    def __init__(self, position: np.ndarray, config: SimulationConfig):
        super().__init__(position, config.pursuer_max_speed, config.world_size)
        self.sensing_radius = config.pursuer_sensing_radius
        self.capture_radius = config.pursuer_capture_radius
        self.inertia = config.pursuer_inertia
        self.cognitive = config.pursuer_cognitive
        self.social = config.pursuer_social

        # PSO memory
        self.personal_best_pos = position.copy()
        self.personal_best_distance = float('inf')
        self.target_evader_ref = None  # Track which evader we're targeting

    def invalidate_if_target_captured(self):
        """Reset personal best if the tracked target evader was captured"""
        if self.target_evader_ref is not None and self.target_evader_ref.captured:
            # Target was captured - reset memory to avoid sticking to that location
            self.personal_best_pos = self.position.copy()
            self.personal_best_distance = float('inf')
            self.target_evader_ref = None

    def update(self, evaders: List['Evader'], pursuers: List['Pursuer'], obstacles: List):
        """Update pursuer using PSO"""
        # Invalidate target if it was captured
        self.invalidate_if_target_captured()
        
        if len(evaders) == 0:
            return

        # Find nearest active evader
        min_dist = float('inf')
        nearest_evader = None
        for evader in evaders:
            if not evader.captured:
                dist = np.linalg.norm(self.position - evader.position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_evader = evader

        if nearest_evader is None:
            return
        
        # Track current target
        self.target_evader_ref = nearest_evader

        # Update personal best
        if min_dist < self.personal_best_distance:
            self.personal_best_distance = min_dist
            self.personal_best_pos = nearest_evader.position.copy()

        # Find global best among nearby pursuers (only if their targets are still alive)
        global_best_pos = self.personal_best_pos.copy()
        global_best_dist = self.personal_best_distance

        for pursuer in pursuers:
            if pursuer != self:
                dist_to_pursuer = np.linalg.norm(self.position - pursuer.position)
                if dist_to_pursuer < self.sensing_radius:
                    # Only consider neighbor's best if their target is still active
                    if (pursuer.target_evader_ref is None or 
                        not pursuer.target_evader_ref.captured):
                        if pursuer.personal_best_distance < global_best_dist:
                            global_best_dist = pursuer.personal_best_distance
                            global_best_pos = pursuer.personal_best_pos.copy()

        # PSO velocity update
        r1, r2 = np.random.rand(2)
        cognitive_component = self.cognitive * r1 * (self.personal_best_pos - self.position)
        social_component = self.social * r2 * (global_best_pos - self.position)

        self.velocity = (self.inertia * self.velocity +
                         cognitive_component +
                         social_component)

        # Obstacle avoidance
        self.avoid_obstacles(obstacles)

        self.limit_speed()
        self.position += self.velocity
        self.obstacle_collision(obstacles)
        self.apply_boundary()

    def avoid_obstacles(self, obstacles: List):
        """Simple obstacle avoidance"""
        if not obstacles:
            return

        avoidance_force = np.zeros(2)
        for obs in obstacles:
            ox, oy, ow, oh = obs
            # Check if agent is near obstacle
            if (ox - 5 < self.position[0] < ox + ow + 5 and
                    oy - 5 < self.position[1] < oy + oh + 5):
                # Calculate repulsion from obstacle center
                obs_center = np.array([ox + ow / 2, oy + oh / 2])
                diff = self.position - obs_center
                dist = np.linalg.norm(diff)
                if dist > 0:
                    avoidance_force += (diff / dist) * 5.0

        self.velocity += avoidance_force


class PSOEvader(Agent):
    """PSO-based evader agent (flees from pursuers)"""

    def __init__(self, position: np.ndarray, config: SimulationConfig):
        super().__init__(position, config.evader_max_speed, config.world_size)
        self.sensing_radius = config.evader_sensing_radius
        self.inertia = config.evader_inertia
        self.cognitive = config.evader_cognitive
        self.social = config.evader_social

        # PSO memory (maximizing distance from pursuers)
        self.personal_best_pos = position.copy()
        self.personal_best_distance = 0.0

    def update(self, pursuers: List[Pursuer], evaders: List['Evader'], obstacles: List):
        """Update evader using PSO (inverse - maximize distance)"""
        if self.captured:
            return

        # Find nearest pursuer
        min_dist = float('inf')
        nearest_pursuer_pos = None
        for pursuer in pursuers:
            dist = np.linalg.norm(self.position - pursuer.position)
            if dist < min_dist:
                min_dist = dist
                nearest_pursuer_pos = pursuer.position

        if nearest_pursuer_pos is None:
            return

        # Update personal best (farther is better)
        if min_dist > self.personal_best_distance:
            self.personal_best_distance = min_dist
            self.personal_best_pos = self.position.copy()

        # Find global best among nearby evaders
        global_best_pos = self.personal_best_pos.copy()
        global_best_dist = self.personal_best_distance

        for evader in evaders:
            if evader != self and not evader.captured:
                dist_to_evader = np.linalg.norm(self.position - evader.position)
                if dist_to_evader < self.sensing_radius:
                    if evader.personal_best_distance > global_best_dist:
                        global_best_dist = evader.personal_best_distance
                        global_best_pos = evader.personal_best_pos.copy()

        # PSO velocity update
        r1, r2 = np.random.rand(2)
        cognitive_component = self.cognitive * r1 * (self.personal_best_pos - self.position)
        social_component = self.social * r2 * (global_best_pos - self.position)

        # Add repulsion from nearest pursuer
        flee_direction = self.position - nearest_pursuer_pos
        flee_dist = np.linalg.norm(flee_direction)
        if flee_dist > 0:
            flee_force = (flee_direction / flee_dist) * 3.0
        else:
            flee_force = np.zeros(2)

        self.velocity = (self.inertia * self.velocity +
                         cognitive_component +
                         social_component +
                         flee_force)

        # Obstacle avoidance
        self.avoid_obstacles(obstacles)

        self.limit_speed()
        self.position += self.velocity
        self.obstacle_collision(obstacles)
        self.apply_boundary()

    def avoid_obstacles(self, obstacles: List):
        """Simple obstacle avoidance"""
        if not obstacles:
            return

        avoidance_force = np.zeros(2)
        for obs in obstacles:
            ox, oy, ow, oh = obs
            if (ox - 5 < self.position[0] < ox + ow + 5 and
                    oy - 5 < self.position[1] < oy + oh + 5):
                obs_center = np.array([ox + ow / 2, oy + oh / 2])
                diff = self.position - obs_center
                dist = np.linalg.norm(diff)
                if dist > 0:
                    avoidance_force += (diff / dist) * 5.0

        self.velocity += avoidance_force


class ACOEvader(Agent):
    """ACO-based evader agent using repulsive pheromones"""

    def __init__(self, position: np.ndarray, config: SimulationConfig, pheromone_grid):
        super().__init__(position, config.evader_max_speed, config.world_size)
        self.sensing_radius = config.evader_sensing_radius
        self.pheromone_grid = pheromone_grid
        self.grid_resolution = config.aco_grid_resolution
        self.pheromone_deposit = config.aco_pheromone_deposit
        self.danger_weight = config.aco_danger_weight
        
        # Dynamic radius parameters for wide area candidate search
        self.r_min = 5.0  # Minimum search radius
        self.epsilon = 0.1  # Normalization factor
        self.k_scale = 0.3  # Obstacle density scaling factor
        self.obstacle_density = 0.0  # Updated dynamically during simulation
        
        # ACO heuristic parameters
        self.alpha = 1.0  # Pheromone influence (lower pheromone = better for evasion)
        self.beta = 2.0   # Heuristic influence (distance from pursuers)

    def get_dynamic_radius(self):
        """Dynamic radius based on map size and obstacle density - Eq. (6)"""
        a, b = self.world_size
        r = max(self.r_min, (1 / self.epsilon) * np.sqrt(a**2 + b**2) * (1 - self.k_scale * self.obstacle_density))
        return r

    def get_wide_area_candidates(self, obstacles, n_samples=8):
        """Sample candidate positions within dynamic search radius"""
        r = self.get_dynamic_radius()
        candidates = []
        attempts = 0
        while len(candidates) < n_samples and attempts < 50:
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, r)
            candidate = self.position + np.array([np.cos(angle), np.sin(angle)]) * dist
            candidate = np.clip(candidate, 0, self.world_size)
            if not self.is_in_obstacle(candidate, obstacles):
                candidates.append(candidate)
            attempts += 1
        return candidates

    def evasion_heuristic(self, candidate_pos: np.ndarray, pursuers: List[Pursuer]) -> float:
        """Distance-based heuristic — farther from nearest pursuer is better"""
        if len(pursuers) == 0:
            return 0.0
        min_dist = min(np.linalg.norm(candidate_pos - p.position) for p in pursuers)
        return min_dist  # Higher is better for evaders

    def select_next_position(self, candidates: List[np.ndarray], pursuers: List[Pursuer]) -> np.ndarray:
        """ACO-based position selection using pheromone and heuristic"""
        weights = []
        for pos in candidates:
            # Get pheromone level at candidate position
            grid_x = int(np.clip(pos[0] / self.grid_resolution, 0, self.pheromone_grid.shape[0] - 1))
            grid_y = int(np.clip(pos[1] / self.grid_resolution, 0, self.pheromone_grid.shape[1] - 1))
            pheromone = max(self.pheromone_grid[grid_x, grid_y], 1e-6)  # Avoid division by zero
            
            # Get heuristic value (distance from nearest pursuer)
            heuristic = self.evasion_heuristic(pos, pursuers)
            
            # ACO formula: inverse for evasion (low pheromone + high distance = good)
            # weight = heuristic^beta / pheromone^alpha
            weight = (heuristic ** self.beta) / (pheromone ** self.alpha)
            weights.append(weight)
        
        # Convert weights to probabilities
        weights = np.array(weights)
        weights = np.maximum(weights, 1e-6)  # Ensure no zero weights
        probs = weights / weights.sum()
        
        # Probabilistically select a candidate
        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx]
    
    def deposit_pheromone_with_diffusion(self, grid_x, grid_y, amount):
        """Deposit pheromone with topological diffusion to neighbours (Liu et al., 2025, Eq. 15)"""
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx = int(np.clip(grid_x + dx, 0, self.pheromone_grid.shape[0]-1))
                ny = int(np.clip(grid_y + dy, 0, self.pheromone_grid.shape[1]-1))
                dist_to_center = np.sqrt(dx**2 + dy**2)
                
                if dist_to_center == 0:
                    # Full strength at source
                    self.pheromone_grid[nx, ny] += amount
                elif dist_to_center <= np.sqrt(2):  # 8-connected neighbors (distance ≤ 1.414)
                    # Moderate diffusion to immediate neighbors
                    self.pheromone_grid[nx, ny] += amount * 0.5
                elif dist_to_center <= 2:  # Diagonal neighbors at distance 2
                    # Weak diffusion to outer neighbors
                    self.pheromone_grid[nx, ny] += amount * 0.25
                # Beyond distance 2: no diffusion

    def update(self, pursuers: List[Pursuer], evaders: List['ACOEvader'], obstacles: List):
        """Update evader using ACO principles"""
        if self.captured:
            return

        # Deposit pheromones where pursuers are detected
        for pursuer in pursuers:
            dist = np.linalg.norm(self.position - pursuer.position)
            if dist < self.sensing_radius:
                grid_x = int(pursuer.position[0] / self.grid_resolution)
                grid_y = int(pursuer.position[1] / self.grid_resolution)
                grid_x = np.clip(grid_x, 0, self.pheromone_grid.shape[0] - 1)
                grid_y = np.clip(grid_y, 0, self.pheromone_grid.shape[1] - 1)
                if dist > 0:
                    # Deposit with diffusion: stronger deposit for closer pursuers (Liu et al., 2025)
                    amount = self.pheromone_deposit / dist
                    self.deposit_pheromone_with_diffusion(grid_x, grid_y, amount)

        # Evaluate candidate positions using wide area search and ACO heuristics
        candidates = self.get_wide_area_candidates(obstacles, n_samples=8)
        
        if len(candidates) > 0:
            # Use ACO heuristic to select best candidate
            next_pos = self.select_next_position(candidates, pursuers)
            best_direction = next_pos - self.position
        else:
            best_direction = np.zeros(2)

        # Normalize and apply
        if np.linalg.norm(best_direction) > 0:
            best_direction = best_direction / np.linalg.norm(best_direction)

        # Add direct flee component from nearest pursuer
        flee_force = np.zeros(2)
        min_pursuer_dist = float('inf')
        for pursuer in pursuers:
            dist = np.linalg.norm(self.position - pursuer.position)
            if dist < min_pursuer_dist and dist < self.sensing_radius:
                min_pursuer_dist = dist
                flee_direction = self.position - pursuer.position
                if np.linalg.norm(flee_direction) > 0:
                    flee_force = (flee_direction / np.linalg.norm(flee_direction)) * 2.0

        # Combine pheromone guidance with direct flee
        self.velocity = 0.7 * self.velocity + 0.3 * (best_direction * 2.0 + flee_force)

        self.limit_speed()
        self.position += self.velocity
        self.obstacle_collision(obstacles)
        self.apply_boundary()

    def is_in_obstacle(self, pos: np.ndarray, obstacles: List) -> bool:
        """Check if position is inside an obstacle"""
        if not obstacles:
            return False
        for obs in obstacles:
            ox, oy, ow, oh = obs
            if ox < pos[0] < ox + ow and oy < pos[1] < oy + oh:
                return True
        return False


class SwarmSimulation:
    """Main simulation environment"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.step_count = 0
        self.capture_times = []

        # Initialize obstacles
        if config.obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = config.obstacles

        # Initialize pursuers
        self.pursuers = []
        for _ in range(config.n_pursuers):
            pos = np.random.rand(2) * np.array(config.world_size) * 0.3
            self.pursuers.append(Pursuer(pos, config))

        # Initialize evaders
        self.evaders = []
        if config.evader_type == "ACO":
            # Create pheromone grid
            grid_size = (config.world_size[0] // config.aco_grid_resolution + 1,
                         config.world_size[1] // config.aco_grid_resolution + 1)
            self.pheromone_grid = np.zeros(grid_size)

            for _ in range(config.n_evaders):
                pos = (np.random.rand(2) * np.array(config.world_size) * 0.3 +
                       np.array(config.world_size) * 0.7)
                self.evaders.append(ACOEvader(pos, config, self.pheromone_grid))
        else:  # PSO
            for _ in range(config.n_evaders):
                pos = (np.random.rand(2) * np.array(config.world_size) * 0.3 +
                       np.array(config.world_size) * 0.7)
                self.evaders.append(PSOEvader(pos, config))

    def step(self):
        """Execute one simulation step"""
        # Update pursuers
        for pursuer in self.pursuers:
            pursuer.update(self.evaders, self.pursuers, self.obstacles)

        # Update evaders
        for evader in self.evaders:
            evader.update(self.pursuers, self.evaders, self.obstacles)

        # Check for captures
        for evader in self.evaders:
            if not evader.captured:
                for pursuer in self.pursuers:
                    dist = np.linalg.norm(evader.position - pursuer.position)
                    if dist < pursuer.capture_radius:
                        evader.captured = True
                        self.capture_times.append(self.step_count)
                        break

        # Decay pheromones (ACO only)
        if self.config.evader_type == "ACO":
            self.pheromone_grid *= self.config.aco_pheromone_decay

        self.step_count += 1

    def run(self, visualize: bool = True):
        """Run the simulation"""
        if visualize:
            self.run_with_visualization()
        else:
            while self.step_count < self.config.max_steps:
                self.step()
                if all(e.captured for e in self.evaders):
                    break

        return self.get_metrics()

    def get_metrics(self):
        """Calculate simulation metrics"""
        n_captured = sum(1 for e in self.evaders if e.captured)
        capture_rate = n_captured / len(self.evaders)
        avg_survival_time = np.mean(self.capture_times) if self.capture_times else self.config.max_steps

        return {
            'capture_rate': capture_rate,
            'n_captured': n_captured,
            'n_evaders': len(self.evaders),
            'avg_survival_time': avg_survival_time,
            'total_steps': self.step_count,
            'capture_times': self.capture_times
        }

    def run_with_visualization(self):
        """Run simulation with animated visualization"""
        fig, ax = plt.subplots(figsize=(10, 10))

        def init():
            ax.clear()
            ax.set_xlim(0, self.config.world_size[0])
            ax.set_ylim(0, self.config.world_size[1])
            ax.set_aspect('equal')
            ax.set_title(f'Pursuit-Evasion: {self.config.evader_type} Evaders')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Draw obstacles
            for obs in self.obstacles:
                rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                         linewidth=1, edgecolor='black',
                                         facecolor='gray', alpha=0.5)
                ax.add_patch(rect)

            return []

        def update(frame):
            if self.step_count >= self.config.max_steps or all(e.captured for e in self.evaders):
                return []

            self.step()
            ax.clear()
            ax.set_xlim(0, self.config.world_size[0])
            ax.set_ylim(0, self.config.world_size[1])
            ax.set_aspect('equal')

            # Draw obstacles
            for obs in self.obstacles:
                rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                         linewidth=1, edgecolor='black',
                                         facecolor='gray', alpha=0.5)
                ax.add_patch(rect)

            # Draw pheromone grid (ACO only)
            if self.config.evader_type == "ACO" and np.max(self.pheromone_grid) > 0:
                extent = [0, self.config.world_size[0], 0, self.config.world_size[1]]
                ax.imshow(self.pheromone_grid.T, origin='lower', extent=extent,
                          cmap='Reds', alpha=0.3, interpolation='bilinear')

            # Draw pursuers
            pursuer_positions = np.array([p.position for p in self.pursuers])
            ax.scatter(pursuer_positions[:, 0], pursuer_positions[:, 1],
                       c='red', s=100, marker='o', label='Pursuers', edgecolors='darkred')

            # Draw evaders
            active_evaders = [e for e in self.evaders if not e.captured]
            if active_evaders:
                evader_positions = np.array([e.position for e in active_evaders])
                ax.scatter(evader_positions[:, 0], evader_positions[:, 1],
                           c='blue', s=100, marker='s', label='Evaders', edgecolors='darkblue')

            # Draw captured evaders
            captured_evaders = [e for e in self.evaders if e.captured]
            if captured_evaders:
                captured_positions = np.array([e.position for e in captured_evaders])
                ax.scatter(captured_positions[:, 0], captured_positions[:, 1],
                           c='gray', s=50, marker='x', label='Captured', alpha=0.5)

            n_captured = len(captured_evaders)
            ax.set_title(
                f'Step {self.step_count} | Captured: {n_captured}/{len(self.evaders)} | Type: {self.config.evader_type}')
            ax.legend(loc='upper right')

            return []

        anim = FuncAnimation(fig, update, init_func=init, frames=self.config.max_steps,
                             interval=50, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()


def create_scenario(scenario_type: str) -> SimulationConfig:
    """Create predefined scenario configurations"""
    config = SimulationConfig()

    if scenario_type == "open_field":
        config.obstacles = []

    elif scenario_type == "sparse_obstacles":
        config.obstacles = [
            (30, 30, 15, 15),
            (60, 20, 10, 20),
            (20, 70, 20, 10),
            (70, 60, 15, 15)
        ]

    elif scenario_type == "dense_maze":
        config.obstacles = [
            (20, 0, 5, 20),
            (40, 50, 5, 40),
            (60, 0, 5, 40),
            (80, 20, 5, 60),
            (0, 30, 40, 5),
            (55, 50, 40, 5),
            (10, 70, 30, 5),
            (15, 30, 5, 25),
            (55, 70, 15, 5),
            (20, 90, 5, 10),
            (60, 85, 15, 5)
        ]

    return config


def run_experiment(scenario: str, evader_type: str, n_runs: int = 5, visualize_first: bool = True):
    """Run multiple trials and collect statistics"""
    results = []

    for i in range(n_runs):
        print(f"Running trial {i + 1}/{n_runs} - Scenario: {scenario}, Evader: {evader_type}")

        config = create_scenario(scenario)
        config.evader_type = evader_type

        sim = SwarmSimulation(config)

        if i == 0 and visualize_first:
            metrics = sim.run(visualize=True)
        else:
            metrics = sim.run(visualize=False)

        results.append(metrics)
        print(f"  Capture rate: {metrics['capture_rate']:.2%}, Avg survival: {metrics['avg_survival_time']:.1f}")

    # Aggregate results
    avg_capture_rate = np.mean([r['capture_rate'] for r in results])
    avg_survival_time = np.mean([r['avg_survival_time'] for r in results])

    print(f"\n=== Summary for {scenario} with {evader_type} evaders ===")
    print(f"Average capture rate: {avg_capture_rate:.2%}")
    print(f"Average survival time: {avg_survival_time:.1f} steps")

    return results


if __name__ == "__main__":
    print("The Hunters and the Hive - Swarm Pursuit-Evasion Simulation")
    print("=" * 60)

    # Demo: Run one visualization for each scenario and evader type
    print("\n1. Open Field - ACO Evaders")
    config = create_scenario("open_field")
    config.evader_type = "ACO"
    sim = SwarmSimulation(config)
    sim.run(visualize=True)

    print("\n2. Open Field - PSO Evaders")
    config = create_scenario("open_field")
    config.evader_type = "PSO"
    sim = SwarmSimulation(config)
    sim.run(visualize=True)

    # Uncomment to run full experiments
    # run_experiment("open_field", "ACO", n_runs=10, visualize_first=False)
    # run_experiment("sparse_obstacles", "ACO", n_runs=10, visualize_first=False)
    # run_experiment("dense_maze", "ACO", n_runs=10, visualize_first=False)

def run_and_snapshot(sim: SwarmSimulation, snapshot_step: int = 60, filename: str = "snapshot.png") -> None:
    """Run *sim* up to *snapshot_step*, save a PNG, then return.

    The simulation is NOT reset or completed — the caller can continue
    stepping or call sim.get_metrics() afterwards.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Advance to the snapshot step (stop early if all evaders are captured)
    while sim.step_count < snapshot_step:
        sim.step()
        if all(e.captured for e in sim.evaders):
            break

    # ── Render current state ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, sim.config.world_size[0])
    ax.set_ylim(0, sim.config.world_size[1])
    ax.set_aspect('equal')

    # Obstacles
    for obs in sim.obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                  linewidth=1, edgecolor='black',
                                  facecolor='gray', alpha=0.5)
        ax.add_patch(rect)

    # Pheromone grid (ACO only)
    if sim.config.evader_type == "ACO" and np.max(sim.pheromone_grid) > 0:
        extent = [0, sim.config.world_size[0], 0, sim.config.world_size[1]]
        ax.imshow(sim.pheromone_grid.T, origin='lower', extent=extent,
                  cmap='Reds', alpha=0.3, interpolation='bilinear')

    # Pursuers
    pursuer_positions = np.array([p.position for p in sim.pursuers])
    ax.scatter(pursuer_positions[:, 0], pursuer_positions[:, 1],
               c='red', s=100, marker='o', label='Pursuers', edgecolors='darkred')

    # Active evaders
    active_evaders = [e for e in sim.evaders if not e.captured]
    if active_evaders:
        evader_positions = np.array([e.position for e in active_evaders])
        ax.scatter(evader_positions[:, 0], evader_positions[:, 1],
                   c='blue', s=100, marker='s', label='Evaders', edgecolors='darkblue')

    # Captured evaders
    captured_evaders = [e for e in sim.evaders if e.captured]
    if captured_evaders:
        captured_positions = np.array([e.position for e in captured_evaders])
        ax.scatter(captured_positions[:, 0], captured_positions[:, 1],
                   c='gray', s=50, marker='x', label='Captured', alpha=0.5)

    n_captured = len(captured_evaders)
    ax.set_title(f'Step {sim.step_count} | Captured: {n_captured}/{len(sim.evaders)} | Type: {sim.config.evader_type}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)