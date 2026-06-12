import json
import time
import itertools
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

from swarm_pursuit_evasion_3 import SimulationConfig, SwarmSimulation, create_scenario, run_and_snapshot


RESULTS_DIR   = Path("experiment_results")
SNAPSHOTS_DIR = Path("experiment_results/snapshots")
RESULTS_DIR.mkdir(exist_ok=True)
SNAPSHOTS_DIR.mkdir(exist_ok=True)

N_RUNS         = 30   # repetitions per configuration 
SNAPSHOT_STEPS  = [30, 100, 250, 450]   # simulation steps at which to capture snapshots


def run_batch(
    config: SimulationConfig,
    n_runs: int = N_RUNS,
    label: str = "",
    snapshot_label: str = "",
) -> list[dict]:

    results = []
    for i in range(n_runs):
        sim = SwarmSimulation(config)

        if i == 0:
            slug = (snapshot_label or label).replace(" ", "_").replace("/", "-").replace("=", "")
            sim = SwarmSimulation(config)

            for snap_step in SNAPSHOT_STEPS:
                filename = SNAPSHOTS_DIR / f"snapshot_{slug}_step{snap_step}.png"
                print(f"  [{label}] run 01/{n_runs}  [snapshot → {filename.name}]")
                run_and_snapshot(sim, snapshot_step=snap_step, filename=str(filename))

            while sim.step_count < config.max_steps:
                sim.step()
            metrics = sim.get_metrics()
        else:
            metrics = sim.run(visualize=False)

        results.append(metrics)
        print(f"  [{label}] run {i + 1:02d}/{n_runs}  "
              f"capture={metrics['capture_rate']:.2%}  "
              f"survival={metrics['avg_survival_time']:.1f}")
    return results


def aggregate(results: list[dict]) -> dict:
    capture_rates   = [r["capture_rate"]      for r in results]
    survival_times  = [r["avg_survival_time"]  for r in results]
    return {
        "n_runs":             len(results),
        "capture_rate_mean":  float(np.mean(capture_rates)),
        "capture_rate_std":   float(np.std(capture_rates)),
        "survival_mean":      float(np.mean(survival_times)),
        "survival_std":       float(np.std(survival_times)),
        "survival_min":       float(np.min(survival_times)),
        "survival_max":       float(np.max(survival_times)),
    }


def mann_whitney(a: list[dict], b: list[dict], metric: str = "capture_rate") -> dict:
    """Mann-Whitney U test comparing two sets of results on a given metric."""
    vals_a = [r[metric] for r in a]
    vals_b = [r[metric] for r in b]
    stat, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {
        "metric":     metric,
        "U_statistic": float(stat),
        "p_value":     float(p),
        "significant": bool(p < 0.05),
    }


def save_results(name: str, data: dict) -> None:
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  ✓  Results saved to {path}")


def print_header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def print_summary(label: str, agg: dict) -> None:
    print(f"  {label}")
    print(f"    Capture rate : {agg['capture_rate_mean']:.3f} ± {agg['capture_rate_std']:.3f}")
    print(f"    Survival time: {agg['survival_mean']:.1f} ± {agg['survival_std']:.1f}  "
          f"[{agg['survival_min']:.0f} – {agg['survival_max']:.0f}]")


# Experiment 1 – Parameter Sensitivity Analysis

def experiment1_parameter_sensitivity():
    """
    Variables (open-field, 30 runs each):
      • Swarm size   : 3, 5, 10 agents (both pursuers and evaders)
      • Inertia      : 0.4, 0.6, 0.8  (pursuer PSO inertia)
      • Sensing radius: 20, 30, 50 units (pursuer sensing)

    """
    print_header("Experiment 1 – Parameter Sensitivity Analysis")
    t0 = time.time()

    output = {"experiment": "parameter_sensitivity", "timestamp": datetime.now().isoformat()}

    #  1A  Swarm size 
    print("\n[1A] Swarm size sweep (n_pursuers = n_evaders ∈ {3, 5, 10})")
    swarm_sizes = [3, 5, 10]
    size_results = {}

    for evader_type in ("ACO", "PSO"):
        size_results[evader_type] = {}
        for n in swarm_sizes:
            config = create_scenario("open_field")
            config.evader_type   = evader_type
            config.n_pursuers    = n
            config.n_evaders     = n
            label = f"size={n} evader={evader_type}"
            raw = run_batch(config, N_RUNS, label,
                            snapshot_label=f"exp1A_n{n}_{evader_type}")
            size_results[evader_type][str(n)] = {
                "config": {"n_pursuers": n, "n_evaders": n, "evader_type": evader_type},
                "aggregate": aggregate(raw),
                "raw": raw,
            }

    # Statistical tests: within each evader type, compare swarm sizes
    size_tests = {}
    for evader_type in ("ACO", "PSO"):
        size_tests[evader_type] = {}
        for (n1, n2) in itertools.combinations(swarm_sizes, 2):
            a_raw = [r for r in size_results[evader_type][str(n1)]["raw"]]
            b_raw = [r for r in size_results[evader_type][str(n2)]["raw"]]
            key = f"{n1}_vs_{n2}"
            size_tests[evader_type][key] = {
                "capture_rate": mann_whitney(a_raw, b_raw, "capture_rate"),
                "avg_survival_time": mann_whitney(a_raw, b_raw, "avg_survival_time"),
            }

    output["swarm_size"] = {"results": size_results, "statistical_tests": size_tests}

    print("\n  — Swarm size summary —")
    for evader_type in ("ACO", "PSO"):
        for n in swarm_sizes:
            agg = size_results[evader_type][str(n)]["aggregate"]
            print_summary(f"n={n}, {evader_type}", agg)

    #  1B  Inertia weight 
    print("\n[1B] Inertia weight sweep (ω ∈ {0.4, 0.6, 0.8})")
    inertia_values = [0.4, 0.6, 0.8]
    inertia_results = {}

    for evader_type in ("ACO", "PSO"):
        inertia_results[evader_type] = {}
        for w in inertia_values:
            config = create_scenario("open_field")
            config.evader_type      = evader_type
            config.pursuer_inertia  = w
            label = f"ω={w} evader={evader_type}"
            raw = run_batch(config, N_RUNS, label,
                            snapshot_label=f"exp1B_w{w}_{evader_type}")
            inertia_results[evader_type][str(w)] = {
                "config": {"pursuer_inertia": w, "evader_type": evader_type},
                "aggregate": aggregate(raw),
                "raw": raw,
            }

    inertia_tests = {}
    for evader_type in ("ACO", "PSO"):
        inertia_tests[evader_type] = {}
        for (w1, w2) in itertools.combinations(inertia_values, 2):
            a_raw = inertia_results[evader_type][str(w1)]["raw"]
            b_raw = inertia_results[evader_type][str(w2)]["raw"]
            key = f"{w1}_vs_{w2}"
            inertia_tests[evader_type][key] = {
                "capture_rate": mann_whitney(a_raw, b_raw, "capture_rate"),
                "avg_survival_time": mann_whitney(a_raw, b_raw, "avg_survival_time"),
            }

    output["inertia"] = {"results": inertia_results, "statistical_tests": inertia_tests}

    print("\n  — Inertia summary —")
    for evader_type in ("ACO", "PSO"):
        for w in inertia_values:
            agg = inertia_results[evader_type][str(w)]["aggregate"]
            print_summary(f"ω={w}, {evader_type}", agg)

    # 1C  Sensing radius 
    print("\n[1C] Sensing radius sweep (r ∈ {20, 30, 50} units)")
    sensing_radii = [20, 30, 50]
    sensing_results = {}

    for evader_type in ("ACO", "PSO"):
        sensing_results[evader_type] = {}
        for r in sensing_radii:
            config = create_scenario("open_field")
            config.evader_type              = evader_type
            config.pursuer_sensing_radius   = float(r)
            config.evader_sensing_radius    = float(r)
            label = f"r={r} evader={evader_type}"
            raw = run_batch(config, N_RUNS, label,
                            snapshot_label=f"exp1C_r{r}_{evader_type}")
            sensing_results[evader_type][str(r)] = {
                "config": {"sensing_radius": r, "evader_type": evader_type},
                "aggregate": aggregate(raw),
                "raw": raw,
            }

    sensing_tests = {}
    for evader_type in ("ACO", "PSO"):
        sensing_tests[evader_type] = {}
        for (r1, r2) in itertools.combinations(sensing_radii, 2):
            a_raw = sensing_results[evader_type][str(r1)]["raw"]
            b_raw = sensing_results[evader_type][str(r2)]["raw"]
            key = f"{r1}_vs_{r2}"
            sensing_tests[evader_type][key] = {
                "capture_rate": mann_whitney(a_raw, b_raw, "capture_rate"),
                "avg_survival_time": mann_whitney(a_raw, b_raw, "avg_survival_time"),
            }

    output["sensing_radius"] = {"results": sensing_results, "statistical_tests": sensing_tests}

    print("\n  — Sensing radius summary —")
    for evader_type in ("ACO", "PSO"):
        for r in sensing_radii:
            agg = sensing_results[evader_type][str(r)]["aggregate"]
            print_summary(f"r={r}, {evader_type}", agg)

    elapsed = time.time() - t0
    output["elapsed_seconds"] = elapsed
    save_results("experiment1_parameter_sensitivity", output)
    print(f"\n  Experiment 1 complete in {elapsed/60:.1f} min")
    return output


# Experiment 2 – Environmental Complexity

def experiment2_environmental_complexity():
    """
    Scenarios  : open_field, sparse_obstacles, dense_maze
    Evader types: ACO, PSO
    30 runs per scenario × evader type combination.
    """
    print_header("Experiment 2 – Environmental Complexity")
    t0 = time.time()

    scenarios     = ["open_field", "sparse_obstacles", "dense_maze"]
    evader_types  = ["ACO", "PSO"]

    output = {"experiment": "environmental_complexity", "timestamp": datetime.now().isoformat()}
    all_results = {}

    for scenario in scenarios:
        all_results[scenario] = {}
        for evader_type in evader_types:
            config = create_scenario(scenario)
            config.evader_type = evader_type
            label = f"{scenario}/{evader_type}"
            print(f"\n[2] Running: {label}")
            raw = run_batch(config, N_RUNS, label,
                            snapshot_label=f"exp2_{scenario}_{evader_type}")
            all_results[scenario][evader_type] = {
                "config": {
                    "scenario": scenario,
                    "evader_type": evader_type,
                    "n_obstacles": len(config.obstacles or []),
                },
                "aggregate": aggregate(raw),
                "raw": raw,
            }

    # Statistical tests
    # (a) ACO vs PSO within each scenario
    comparison_aco_vs_pso = {}
    for scenario in scenarios:
        a_raw = all_results[scenario]["ACO"]["raw"]
        b_raw = all_results[scenario]["PSO"]["raw"]
        comparison_aco_vs_pso[scenario] = {
            "capture_rate":     mann_whitney(a_raw, b_raw, "capture_rate"),
            "avg_survival_time": mann_whitney(a_raw, b_raw, "avg_survival_time"),
        }

    # (b) Across scenarios within each evader type
    comparison_across_scenarios = {}
    for evader_type in evader_types:
        comparison_across_scenarios[evader_type] = {}
        for (s1, s2) in itertools.combinations(scenarios, 2):
            a_raw = all_results[s1][evader_type]["raw"]
            b_raw = all_results[s2][evader_type]["raw"]
            key = f"{s1}_vs_{s2}"
            comparison_across_scenarios[evader_type][key] = {
                "capture_rate":     mann_whitney(a_raw, b_raw, "capture_rate"),
                "avg_survival_time": mann_whitney(a_raw, b_raw, "avg_survival_time"),
            }

    output["results"]                   = all_results
    output["comparison_aco_vs_pso"]     = comparison_aco_vs_pso
    output["comparison_across_scenarios"] = comparison_across_scenarios

    # Print summary table
    print("\n  — Environmental complexity summary —")
    print(f"  {'Scenario':<20} {'Evader':<6} {'Cap.Rate':>9}  {'Survival':>10}")
    print("  " + "-" * 52)
    for scenario in scenarios:
        for evader_type in evader_types:
            agg = all_results[scenario][evader_type]["aggregate"]
            print(f"  {scenario:<20} {evader_type:<6} "
                  f"{agg['capture_rate_mean']:>7.3f}±{agg['capture_rate_std']:.3f}  "
                  f"{agg['survival_mean']:>7.1f}±{agg['survival_std']:.1f}")

    elapsed = time.time() - t0
    output["elapsed_seconds"] = elapsed
    save_results("experiment2_environmental_complexity", output)
    print(f"\n  Experiment 2 complete in {elapsed/60:.1f} min")
    return output


# Experiment 3 – Scalability Analysis

def experiment3_scalability():
    """
    Pursuer-to-evader ratios: 1:1, 2:1, 1:2
    Applied in all three scenarios and for both evader types.
    """
    print_header("Experiment 3 – Scalability Analysis (pursuer-to-evader ratios)")
    t0 = time.time()

    # Ratios as (n_pursuers, n_evaders)
    ratios = [
        (5, 5,  "1:1"),
        (10, 5, "2:1"),
        (5, 10, "1:2"),
    ]
    scenarios    = ["open_field", "sparse_obstacles", "dense_maze"]
    evader_types = ["ACO", "PSO"]

    output = {"experiment": "scalability", "timestamp": datetime.now().isoformat()}
    all_results = {}

    for scenario in scenarios:
        all_results[scenario] = {}
        for evader_type in evader_types:
            all_results[scenario][evader_type] = {}
            for n_pur, n_eva, ratio_label in ratios:
                config = create_scenario(scenario)
                config.evader_type = evader_type
                config.n_pursuers  = n_pur
                config.n_evaders   = n_eva
                label = f"{scenario}/{evader_type}/{ratio_label}"
                print(f"\n[3] Running: {label}  ({n_pur} vs {n_eva})")
                raw = run_batch(config, N_RUNS, label,
                                snapshot_label=f"exp3_{scenario}_{evader_type}_{ratio_label.replace(':', '-')}")
                all_results[scenario][evader_type][ratio_label] = {
                    "config": {
                        "scenario": scenario,
                        "evader_type": evader_type,
                        "n_pursuers": n_pur,
                        "n_evaders": n_eva,
                        "ratio": ratio_label,
                    },
                    "aggregate": aggregate(raw),
                    "raw": raw,
                }

    # Statistical tests: compare ratios within each scenario × evader_type cell
    ratio_labels = [r[2] for r in ratios]
    stat_tests = {}
    for scenario in scenarios:
        stat_tests[scenario] = {}
        for evader_type in evader_types:
            stat_tests[scenario][evader_type] = {}
            for (r1, r2) in itertools.combinations(ratio_labels, 2):
                a_raw = all_results[scenario][evader_type][r1]["raw"]
                b_raw = all_results[scenario][evader_type][r2]["raw"]
                key = f"{r1}_vs_{r2}"
                stat_tests[scenario][evader_type][key] = {
                    "capture_rate":     mann_whitney(a_raw, b_raw, "capture_rate"),
                    "avg_survival_time": mann_whitney(a_raw, b_raw, "avg_survival_time"),
                }

    output["results"]         = all_results
    output["statistical_tests"] = stat_tests

    # Summary table
    print("\n  — Scalability summary —")
    print(f"  {'Scenario':<20} {'Evader':<6} {'Ratio':<5} {'Cap.Rate':>9}  {'Survival':>10}")
    print("  " + "-" * 60)
    for scenario in scenarios:
        for evader_type in evader_types:
            for _, _, ratio_label in ratios:
                agg = all_results[scenario][evader_type][ratio_label]["aggregate"]
                print(f"  {scenario:<20} {evader_type:<6} {ratio_label:<5} "
                      f"{agg['capture_rate_mean']:>7.3f}±{agg['capture_rate_std']:.3f}  "
                      f"{agg['survival_mean']:>7.1f}±{agg['survival_std']:.1f}")

    elapsed = time.time() - t0
    output["elapsed_seconds"] = elapsed
    save_results("experiment3_scalability", output)
    print(f"\n  Experiment 3 complete in {elapsed/60:.1f} min")
    return output


# Main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run planned experiments for 'The Hunters and the Hive'."
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=[1, 2, 3],
        help="Which experiments to run, e.g. --experiments 1 2  (default: all)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=N_RUNS,
        help=f"Replications per configuration (default: {N_RUNS})"
    )
    parser.add_argument(
        "--snapshot-step", type=int, default=SNAPSHOT_STEPS,
        help=f"Simulation steps at which to capture snapshots (default: {SNAPSHOT_STEPS})"
    )
    args = parser.parse_args()
    N_RUNS        = args.n_runs        # override global for quick testing
    SNAPSHOT_STEPS = args.snapshot_step  # override global snapshot timing

    print("The Hunters and the Hive – Planned Experiments")
    print(f"Running {N_RUNS} replications per configuration")
    print(f"Results will be saved to ./{RESULTS_DIR}/")

    if 1 in args.experiments:
        experiment1_parameter_sensitivity()
    if 2 in args.experiments:
        experiment2_environmental_complexity()
    if 3 in args.experiments:
        experiment3_scalability()

    print("\n\nAll requested experiments complete.")