"""Parallelized, Resumable Dev-Set Ablation Study for CADM-MSWN and Reference Baselines.

Strictly executed on the Dev Split (120 pairs):
    - 100 MSRS train pairs (random seed 0)
    - 5 Harvard CT-MRI held-out pairs
    - 5 Harvard PET-MRI held-out pairs
    - 5 Harvard SPECT-MRI held-out pairs
    - 5 Lytro held-out pairs

Optimizations:
    1. Resumable execution: skips configurations present in results/dev_ablation_summary.csv.
    2. Per-pair on-disk caching of diffusion (T=5, 10, 20 from a single pass),
       cluster masks (K=1, 2, 3), and base fused components.
    3. Multiprocessing with 6 workers (seeded & deterministic).
    4. Dual logging to stdout and results/dev_sweep.log.
    5. Saves results to results/dev_ablation_summary.csv and results/dev_ablation_per_pair.csv.
    6. Produces final ranking according to selection_rule.md.
"""

import os
import sys
import time
from pathlib import Path
import yaml
import multiprocessing as mp
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dev_ablation_utils import (
    init_worker,
    build_pair_cache,
    evaluate_pair_worker,
)


class TeeLogger:
    """Tee logger writing simultaneously to console and log file with immediate flush."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def load_dev_pairs():
    """Load the 120 dev pairs defined in splits.yaml."""
    with open("splits.yaml", "r") as f:
        splits = yaml.safe_load(f)

    dev_data = splits["dev"]
    pairs = []

    # 1. MSRS
    for item in dev_data["msrs"]:
        pairs.append({
            "category": "MSRS (IVIF)",
            "pair_id": item["id"],
            "path1": item["ir"],
            "path2": item["vi"],
        })

    # 2. Harvard CT-MRI
    for item in dev_data["harvard_ct_mri"]:
        pairs.append({
            "category": "Harvard (CT-MRI)",
            "pair_id": item["id"],
            "path1": item["src1"],
            "path2": item["src2"],
        })

    # 3. Harvard PET-MRI
    for item in dev_data["harvard_pet_mri"]:
        pairs.append({
            "category": "Harvard (PET-MRI)",
            "pair_id": item["id"],
            "path1": item["src1"],
            "path2": item["src2"],
        })

    # 4. Harvard SPECT-MRI
    for item in dev_data["harvard_spect_mri"]:
        pairs.append({
            "category": "Harvard (SPECT-MRI)",
            "pair_id": item["id"],
            "path1": item["src1"],
            "path2": item["src2"],
        })

    # 5. Lytro
    for item in dev_data["lytro"]:
        pairs.append({
            "category": "Lytro (Multi-focus)",
            "pair_id": item["id"],
            "path1": item["src1"],
            "path2": item["src2"],
        })

    return pairs


def get_ablation_configs():
    """Returns the 12 CADM configurations + 2 Reference Baselines."""
    return [
        # --- 1. Edge Compensation Ablation ---
        {
            "name": "CADM_v1_prototype (gamma=0.5)",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.5, "norm": "linear", "temp": 0.1, "radius": 5,
            "desc": "Baseline prototype: T=20, K=3, gamma=0.5 with edge compensation"
        },
        {
            "name": "gamma_0.25",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.25, "norm": "linear", "temp": 0.1, "radius": 5,
            "desc": "Intermediate edge compensation: gamma=0.25"
        },
        {
            "name": "gamma_0.0_no_edge_comp",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5,
            "desc": "Zero edge compensation anchor: pure GT + FT layered reconstruction"
        },

        # --- 2. Weight Smoothing Ablations (gamma=0) ---
        {
            "name": "softmax_tau_0.1",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.0, "norm": "softmax", "temp": 0.1, "radius": 5,
            "desc": "Softmax normalization with sharp temperature tau=0.1"
        },
        {
            "name": "softmax_tau_0.5",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.0, "norm": "softmax", "temp": 0.5, "radius": 5,
            "desc": "Softmax normalization with smooth temperature tau=0.5"
        },
        {
            "name": "softmax_tau_1.0",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.0, "norm": "softmax", "temp": 1.0, "radius": 5,
            "desc": "Softmax normalization with very smooth temperature tau=1.0"
        },
        {
            "name": "guided_radius_9",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 9,
            "desc": "Larger guided filter radius r=9 (19x19 window) for spatial smoothness"
        },
        {
            "name": "guided_radius_13",
            "type": "cadm",
            "T": 20, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 13,
            "desc": "Wide guided filter radius r=13 (27x27 window)"
        },

        # --- 3. Diffusion Iterations T Ablations (gamma=0) ---
        {
            "name": "T_5",
            "type": "cadm",
            "T": 5, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5,
            "desc": "Low diffusion scale separation: T=5"
        },
        {
            "name": "T_10",
            "type": "cadm",
            "T": 10, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5,
            "desc": "Medium diffusion scale separation: T=10"
        },

        # --- 4. Cluster Count K Ablations (gamma=0) ---
        {
            "name": "K_1_no_clustering",
            "type": "cadm",
            "T": 20, "K": 1, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5,
            "desc": "Global unclustered weighting: K=1"
        },
        {
            "name": "K_2",
            "type": "cadm",
            "T": 20, "K": 2, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5,
            "desc": "Two-regime clustering: K=2"
        },

        # --- 5. Reference Baselines ---
        {
            "name": "Averaging",
            "type": "baseline_avg",
            "desc": "Pixel-wise mean baseline"
        },
        {
            "name": "GFF",
            "type": "baseline_gff",
            "desc": "Guided Filter Fusion (Li et al., 2013)"
        },
    ]


def compute_final_ranking(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank all configurations according to selection_rule.md Section 1.2 and 1.3."""
    df = df_summary.copy()

    # Primary metrics: QABF, NABF (lower is better), MI, SSIM, VIFP
    # Note: method='min' assigns rank 1 to best value
    df["Rank_QABF"] = df["Mean_QABF"].rank(ascending=False, method="min")
    df["Rank_NABF"] = df["Mean_NABF"].rank(ascending=True, method="min")
    df["Rank_MI"] = df["Mean_MI"].rank(ascending=False, method="min")
    df["Rank_SSIM"] = df["Mean_SSIM"].rank(ascending=False, method="min")
    df["Rank_VIFP"] = df["Mean_VIFP"].rank(ascending=False, method="min")

    df["Mean_Rank"] = (
        df["Rank_QABF"] + df["Rank_NABF"] + df["Rank_MI"] + df["Rank_SSIM"] + df["Rank_VIFP"]
    ) / 5.0

    # Sort by Mean_Rank ascending; tie-break on Mean_Latency_ms ascending
    df_sorted = df.sort_values(
        by=["Mean_Rank", "Mean_Latency_ms"],
        ascending=[True, True]
    ).reset_index(drop=True)

    df_sorted["Overall_Rank"] = range(1, len(df_sorted) + 1)
    return df_sorted


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "dev_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "dev_sweep.log"
    sys.stdout = TeeLogger(log_path)

    print("==================================================================", flush=True)
    print(f"CADM-MSWN DEV-SET ABLATION SWEEP (6 WORKERS, DETACHED RUN)", flush=True)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("==================================================================", flush=True)

    # 1. Load dev pairs
    dev_pairs = load_dev_pairs()
    print(f"Total dev pairs loaded: {len(dev_pairs)}", flush=True)

    # 2. Check finished configurations for resumability
    summary_file = out_dir / "dev_ablation_summary.csv"
    per_pair_file = out_dir / "dev_ablation_per_pair.csv"
    compat_results_file = out_dir / "dev_ablation_results.csv"

    finished_configs = set()
    summary_records = []
    if summary_file.exists():
        df_existing = pd.read_csv(summary_file)
        summary_records = df_existing.to_dict("records")
        finished_configs = set(df_existing["Config"].tolist())
        print(f"Found {len(finished_configs)} already finished configuration(s) in {summary_file}:")
        for fc in finished_configs:
            print(f"  - [FINISHED] {fc}")
    else:
        print("No prior summary found. Starting clean run.")

    # Synchronize per_pair files
    all_per_pair_records = []
    if per_pair_file.exists():
        all_per_pair_records = pd.read_csv(per_pair_file).to_dict("records")
    elif compat_results_file.exists():
        all_per_pair_records = pd.read_csv(compat_results_file).to_dict("records")
        pd.DataFrame(all_per_pair_records).to_csv(per_pair_file, index=False)

    # 3. Setup multiprocessing pool (6 workers)
    num_workers = 6
    print(f"\nInitializing multiprocessing pool with {num_workers} workers...", flush=True)
    pool = mp.Pool(processes=num_workers, initializer=init_worker)

    # 4. Cache precomputations across all pairs
    print(f"\nVerifying / building disk intermediate cache in {cache_dir}...", flush=True)
    t_cache_start = time.perf_counter()
    cache_tasks = [(p, str(cache_dir)) for p in dev_pairs]
    # Build cache in parallel
    cached_ids = pool.starmap(build_pair_cache, cache_tasks)
    t_cache_elapsed = time.perf_counter() - t_cache_start
    print(f"Disk cache ready for all {len(cached_ids)} pairs in {t_cache_elapsed:.2f}s.\n", flush=True)

    # 5. Run sweep over configurations
    configs = get_ablation_configs()
    print(f"Total configurations in experimental pool: {len(configs)}", flush=True)

    for cfg_dict in configs:
        cfg_name = cfg_dict["name"]

        # Check if already finished
        if cfg_name in finished_configs:
            print(f"\n>>> Skipping finished configuration: {cfg_name}", flush=True)
            continue

        print(f"\n==========================================", flush=True)
        print(f"Running: {cfg_name} ({cfg_dict['desc']})", flush=True)
        print(f"==========================================", flush=True)

        t_start = time.perf_counter()
        eval_tasks = [(cfg_dict, p, str(cache_dir)) for p in dev_pairs]
        pair_records = pool.map(evaluate_pair_worker, eval_tasks)
        total_wall_time = time.perf_counter() - t_start

        # Record per-pair results
        all_per_pair_records.extend(pair_records)
        df_cfg = pd.DataFrame(pair_records)

        # Compute summary
        summary = {
            "Config": cfg_name,
            "Type": cfg_dict["type"],
            "Mean_QABF": round(float(df_cfg["QABF"].mean()), 4),
            "Std_QABF": round(float(df_cfg["QABF"].std()), 4),
            "Mean_NABF": round(float(df_cfg["NABF"].mean()), 4),
            "Std_NABF": round(float(df_cfg["NABF"].std()), 4),
            "Mean_MI": round(float(df_cfg["MI"].mean()), 4),
            "Std_MI": round(float(df_cfg["MI"].std()), 4),
            "Mean_SSIM": round(float(df_cfg["SSIM"].mean()), 4),
            "Std_SSIM": round(float(df_cfg["SSIM"].std()), 4),
            "Mean_VIFP": round(float(df_cfg["VIFP"].mean()), 4),
            "Std_VIFP": round(float(df_cfg["VIFP"].std()), 4),
            "Mean_EN": round(float(df_cfg["EN"].mean()), 4),
            "Std_EN": round(float(df_cfg["EN"].std()), 4),
            "Mean_SD": round(float(df_cfg["SD"].mean()), 4),
            "Std_SD": round(float(df_cfg["SD"].std()), 4),
            "Mean_SF": round(float(df_cfg["SF"].mean()), 4),
            "Std_SF": round(float(df_cfg["SF"].std()), 4),
            "Mean_AG": round(float(df_cfg["AG"].mean()), 4),
            "Std_AG": round(float(df_cfg["AG"].std()), 4),
            "Mean_Latency_ms": round(float(df_cfg["Latency_ms"].mean()), 2),
            "Std_Latency_ms": round(float(df_cfg["Latency_ms"].std()), 2),
            "Total_Wall_Time_s": round(total_wall_time, 2),
        }
        summary_records.append(summary)
        finished_configs.add(cfg_name)

        # Save immediately to ensure resumability
        pd.DataFrame(all_per_pair_records).to_csv(per_pair_file, index=False)
        pd.DataFrame(all_per_pair_records).to_csv(compat_results_file, index=False)
        pd.DataFrame(summary_records).to_csv(summary_file, index=False)

        print(f"[{cfg_name}] Completed in {total_wall_time:.1f}s | "
              f"QABF: {summary['Mean_QABF']}±{summary['Std_QABF']} | "
              f"NABF: {summary['Mean_NABF']}±{summary['Std_NABF']} | "
              f"MI: {summary['Mean_MI']}±{summary['Std_MI']} | "
              f"SSIM: {summary['Mean_SSIM']}±{summary['Std_SSIM']} | "
              f"VIFP: {summary['Mean_VIFP']}±{summary['Std_VIFP']}", flush=True)

    pool.close()
    pool.join()

    # 6. Compute and report final ranking according to selection_rule.md
    df_final_summary = pd.DataFrame(summary_records)
    df_ranked = compute_final_ranking(df_final_summary)
    ranked_file = out_dir / "dev_ablation_ranked.csv"
    df_ranked.to_csv(ranked_file, index=False)

    print("\n==================================================================", flush=True)
    print("=== FINAL DEV-SET ABLATION & BASELINE RANKING TABLE ===", flush=True)
    print("==================================================================", flush=True)
    cols_to_show = [
        "Overall_Rank", "Config", "Mean_Rank",
        "Mean_QABF", "Mean_NABF", "Mean_MI", "Mean_SSIM",
        "Mean_VIFP", "Mean_EN", "Mean_SD", "Mean_SF", "Mean_AG",
        "Mean_Latency_ms", "Total_Wall_Time_s"
    ]
    print(df_ranked[cols_to_show].to_string(index=False), flush=True)
    print("\n=== COMPLETE DEV-SET SWEEP FINISHED SUCCESSFULLY ===", flush=True)


if __name__ == "__main__":
    main()
