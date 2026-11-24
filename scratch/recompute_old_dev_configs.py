import os
import sys
import time
from pathlib import Path
import yaml
import multiprocessing as mp
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dev_ablation_utils import (
    init_worker,
    evaluate_pair_worker,
)
from experiments.run_dev_ablations import (
    load_dev_pairs,
    compute_final_ranking,
)

TARGET_CONFIGS = [
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
    {
        "name": "softmax_tau_0.1",
        "type": "cadm",
        "T": 20, "K": 3, "gamma": 0.0, "norm": "softmax", "temp": 0.1, "radius": 5,
        "desc": "Softmax normalization with sharp temperature tau=0.1"
    },
]

def main():
    print("=== RECOMPUTING 4 EARLY DEV CONFIGURATIONS WITH CANONICAL METRICS ===")
    dev_pairs = load_dev_pairs()
    cache_dir = Path("results/dev_cache")
    per_pair_file = Path("results/dev_ablation_per_pair.csv")
    summary_file = Path("results/dev_ablation_summary.csv")

    pool = mp.Pool(processes=6, initializer=init_worker)
    new_per_pair = []
    new_summaries = []

    for cfg_dict in TARGET_CONFIGS:
        cfg_name = cfg_dict["name"]
        print(f"\nRecomputing: {cfg_name}...")
        t_start = time.perf_counter()
        eval_tasks = [(cfg_dict, p, str(cache_dir)) for p in dev_pairs]
        pair_records = pool.map(evaluate_pair_worker, eval_tasks)
        total_wall_time = time.perf_counter() - t_start

        for r in pair_records:
            r["metric_version"] = "canonical_v2_xydeas2000"
            new_per_pair.append(r)

        df_cfg = pd.DataFrame(pair_records)
        summary = {
            "Config": cfg_name,
            "Type": cfg_dict["type"],
            "metric_version": "canonical_v2_xydeas2000",
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
        new_summaries.append(summary)
        print(f"  Done in {total_wall_time:.1f}s | QABF: {summary['Mean_QABF']}, NABF: {summary['Mean_NABF']}")

    pool.close()
    pool.join()

    # Update per-pair CSV
    if per_pair_file.exists():
        df_old_pp = pd.read_csv(per_pair_file)
        if "metric_version" not in df_old_pp.columns:
            df_old_pp["metric_version"] = "canonical_v2_xydeas2000"
        # Filter out old target configs
        target_names = {c["name"] for c in TARGET_CONFIGS}
        df_kept = df_old_pp[~df_old_pp["Config"].isin(target_names)]
        df_new_pp = pd.DataFrame(new_per_pair)
        df_combined_pp = pd.concat([df_kept, df_new_pp], ignore_index=True)
        # Drop VIFF if exists
        if "VIFF" in df_combined_pp.columns:
            df_combined_pp = df_combined_pp.drop(columns=["VIFF"])
        df_combined_pp.to_csv(per_pair_file, index=False)
        print(f"\nUpdated {per_pair_file} with {len(df_combined_pp)} rows.")

    # Update summary CSV
    if summary_file.exists():
        df_old_sum = pd.read_csv(summary_file)
        if "metric_version" not in df_old_sum.columns:
            df_old_sum["metric_version"] = "canonical_v2_xydeas2000"
        target_names = {c["name"] for c in TARGET_CONFIGS}
        df_kept_sum = df_old_sum[~df_old_sum["Config"].isin(target_names)]
        df_new_sum = pd.DataFrame(new_summaries)
        df_combined_sum = pd.concat([df_kept_sum, df_new_sum], ignore_index=True)
        if "Mean_VIFF" in df_combined_sum.columns:
            df_combined_sum = df_combined_sum.drop(columns=["Mean_VIFF", "Std_VIFF"])
        df_combined_sum.to_csv(summary_file, index=False)
        print(f"Updated {summary_file} with {len(df_combined_sum)} rows.")

        # Re-rank
        df_ranked = compute_final_ranking(df_combined_sum)
        df_ranked.to_csv("results/dev_ablation_ranked.csv", index=False)
        print("Updated results/dev_ablation_ranked.csv")

if __name__ == "__main__":
    main()
