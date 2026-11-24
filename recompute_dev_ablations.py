import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb
from fusion.weighting import fuse_cluster_components, reconstruct_fused_image
from baselines.averaging import fuse_averaging
from baselines.gff import fuse_gff
from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import compute_standard_deviation, compute_spatial_frequency, compute_average_gradient
from metrics.structural import compute_qabf_and_nabf, compute_ssim, compute_vif

def load_dev_pairs():
    with open("splits.yaml", "r") as f:
        splits = yaml.safe_load(f)
    dev_data = splits["dev"]
    pairs = []
    for item in dev_data["msrs"]:
        pairs.append({"category": "MSRS (IVIF)", "pair_id": item["id"], "path1": item["ir"], "path2": item["vi"]})
    for item in dev_data["harvard_ct_mri"]:
        pairs.append({"category": "Harvard (CT-MRI)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})
    for item in dev_data["harvard_pet_mri"]:
        pairs.append({"category": "Harvard (PET-MRI)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})
    for item in dev_data["harvard_spect_mri"]:
        pairs.append({"category": "Harvard (SPECT-MRI)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})
    for item in dev_data["lytro"]:
        pairs.append({"category": "Lytro (Multi-focus)", "pair_id": item["id"], "path1": item["src1"], "path2": item["src2"]})
    return pairs

def get_ablation_configs():
    return [
        {"name": "CADM_v1_prototype (gamma=0.5)", "type": "cadm", "T": 20, "K": 3, "gamma": 0.5, "norm": "linear", "temp": 0.1, "radius": 5},
        {"name": "gamma_0.25", "type": "cadm", "T": 20, "K": 3, "gamma": 0.25, "norm": "linear", "temp": 0.1, "radius": 5},
        {"name": "gamma_0.0_no_edge_comp", "type": "cadm", "T": 20, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5},
        {"name": "softmax_tau_0.1", "type": "cadm", "T": 20, "K": 3, "gamma": 0.0, "norm": "softmax", "temp": 0.1, "radius": 5},
        {"name": "softmax_tau_0.5", "type": "cadm", "T": 20, "K": 3, "gamma": 0.0, "norm": "softmax", "temp": 0.5, "radius": 5},
        {"name": "softmax_tau_1.0", "type": "cadm", "T": 20, "K": 3, "gamma": 0.0, "norm": "softmax", "temp": 1.0, "radius": 5},
        {"name": "guided_radius_9", "type": "cadm", "T": 20, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 9},
        {"name": "guided_radius_13", "type": "cadm", "T": 20, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 13},
        {"name": "T_5", "type": "cadm", "T": 5, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5},
        {"name": "T_10", "type": "cadm", "T": 10, "K": 3, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5},
        {"name": "K_1_gamma_0.0", "type": "cadm", "T": 20, "K": 1, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5},
        {"name": "K_2", "type": "cadm", "T": 20, "K": 2, "gamma": 0.0, "norm": "linear", "temp": 0.1, "radius": 5},
        {"name": "Averaging", "type": "baseline_avg"},
        {"name": "GFF", "type": "baseline_gff"},
    ]

def evaluate_pair(cfg, pair_data, cache_dir):
    p_id = pair_data["pair_id"]
    cache_path = os.path.join(cache_dir, f"{p_id}.npz")
    cache = np.load(cache_path)

    im1 = cache["im1"]
    im2 = cache["im2"]
    lum1 = cache["lum1"]
    lum2 = cache["lum2"]
    is_color = bool(cache["is_color"])
    target_cb = cache["target_cb"]
    target_cr = cache["target_cr"]

    t0 = time.perf_counter()

    if cfg["type"] == "baseline_avg":
        fused = fuse_averaging(im1, im2)
    elif cfg["type"] == "baseline_gff":
        fused = fuse_gff(im1, im2)
    else:
        cfg_name = cfg["name"]
        if cfg_name in ("CADM_v1_prototype (gamma=0.5)", "gamma_0.25", "gamma_0.0_no_edge_comp"):
            f_lum = reconstruct_fused_image(
                cache["fused_gt_base"], cache["fused_ge_base"],
                cache["fused_ft_base"], cache["fused_fe_base"],
                alpha=1.0, beta=1.0, gamma=cfg["gamma"]
            )
        elif cfg_name.startswith("softmax_tau_"):
            c1 = (cache["gt1_20"], cache["ge1_20"], cache["ft1_20"], cache["fe1_20"])
            c2 = (cache["gt2_20"], cache["ge2_20"], cache["ft2_20"], cache["fe2_20"])
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, cache["m1_k3"], cache["m2_k3"], lum1, lum2,
                window_size=7, normalization="softmax", temperature=cfg["temp"],
                guided_radius=5, guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(f_gt, f_ge, f_ft, f_fe, alpha=1.0, beta=1.0, gamma=0.0)
        elif cfg_name.startswith("guided_radius_"):
            c1 = (cache["gt1_20"], cache["ge1_20"], cache["ft1_20"], cache["fe1_20"])
            c2 = (cache["gt2_20"], cache["ge2_20"], cache["ft2_20"], cache["fe2_20"])
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, cache["m1_k3"], cache["m2_k3"], lum1, lum2,
                window_size=7, normalization="linear", temperature=0.1,
                guided_radius=cfg["radius"], guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(f_gt, f_ge, f_ft, f_fe, alpha=1.0, beta=1.0, gamma=0.0)
        elif cfg_name.startswith("T_"):
            T = cfg["T"]
            c1 = (cache[f"gt1_{T}"], cache[f"ge1_{T}"], cache[f"ft1_{T}"], cache[f"fe1_{T}"])
            c2 = (cache[f"gt2_{T}"], cache[f"ge2_{T}"], cache[f"ft2_{T}"], cache[f"fe2_{T}"])
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, cache["m1_k3"], cache["m2_k3"], lum1, lum2,
                window_size=7, normalization="linear", temperature=0.1,
                guided_radius=5, guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(f_gt, f_ge, f_ft, f_fe, alpha=1.0, beta=1.0, gamma=0.0)
        elif cfg_name.startswith("K_"):
            K = cfg["K"]
            c1 = (cache["gt1_20"], cache["ge1_20"], cache["ft1_20"], cache["fe1_20"])
            c2 = (cache["gt2_20"], cache["ge2_20"], cache["ft2_20"], cache["fe2_20"])
            f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
                c1, c2, cache[f"m1_k{K}"], cache[f"m2_k{K}"], lum1, lum2,
                window_size=7, normalization="linear", temperature=0.1,
                guided_radius=5, guided_eps=0.0001
            )
            f_lum = reconstruct_fused_image(f_gt, f_ge, f_ft, f_fe, alpha=1.0, beta=1.0, gamma=0.0)

        if is_color:
            fused = ycbcr_to_rgb(f_lum, target_cb, target_cr)
        else:
            fused = f_lum

    lat_ms = (time.perf_counter() - t0) * 1000.0

    qabf, nabf = compute_qabf_and_nabf(fused, im1, im2)
    mi = compute_mutual_information(fused, im1, im2)
    ssim = compute_ssim(fused, im1, im2)
    vifp = compute_vif(fused, im1, im2)
    en = compute_entropy(fused)
    sd = compute_standard_deviation(fused)
    sf = compute_spatial_frequency(fused)
    ag = compute_average_gradient(fused)

    return {
        "Config": cfg["name"],
        "Category": pair_data["category"],
        "Pair": p_id,
        "Latency_ms": round(lat_ms, 2),
        "QABF": qabf,
        "NABF": nabf,
        "MI": mi,
        "SSIM": ssim,
        "VIFP": vifp,
        "EN": en,
        "SD": sd,
        "SF": sf,
        "AG": ag,
    }

def main():
    cache_dir = "results/dev_cache"
    dev_pairs = load_dev_pairs()
    configs = get_ablation_configs()
    print(f"Loaded {len(dev_pairs)} dev pairs. Evaluating {len(configs)} configurations...")

    all_per_pair = []
    summary_records = []

    for cfg in configs:
        c_name = cfg["name"]
        t_start = time.time()
        print(f"Processing {c_name} ...", end="", flush=True)

        c_results = []
        for p in dev_pairs:
            res = evaluate_pair(cfg, p, cache_dir)
            c_results.append(res)
            all_per_pair.append(res)

        wall_time = time.time() - t_start
        df_c = pd.DataFrame(c_results)

        rec = {
            "Config": c_name,
            "Type": cfg["type"],
            "Mean_QABF": round(df_c["QABF"].mean(), 4),
            "Std_QABF": round(df_c["QABF"].std(), 4),
            "Mean_NABF": round(df_c["NABF"].mean(), 4),
            "Std_NABF": round(df_c["NABF"].std(), 4),
            "Mean_MI": round(df_c["MI"].mean(), 4),
            "Std_MI": round(df_c["MI"].std(), 4),
            "Mean_SSIM": round(df_c["SSIM"].mean(), 4),
            "Std_SSIM": round(df_c["SSIM"].std(), 4),
            "Mean_VIFP": round(df_c["VIFP"].mean(), 4),
            "Std_VIFP": round(df_c["VIFP"].std(), 4),
            "Mean_EN": round(df_c["EN"].mean(), 4),
            "Std_EN": round(df_c["EN"].std(), 4),
            "Mean_SD": round(df_c["SD"].mean(), 4),
            "Std_SD": round(df_c["SD"].std(), 4),
            "Mean_SF": round(df_c["SF"].mean(), 4),
            "Std_SF": round(df_c["SF"].std(), 4),
            "Mean_AG": round(df_c["AG"].mean(), 4),
            "Std_AG": round(df_c["AG"].std(), 4),
            "Mean_Latency_ms": round(df_c["Latency_ms"].mean(), 2),
            "Std_Latency_ms": round(df_c["Latency_ms"].std(), 2),
            "Total_Wall_Time_s": round(wall_time, 2),
        }
        summary_records.append(rec)
        print(f" done in {wall_time:.1f}s | QABF: {rec['Mean_QABF']:.4f}, SSIM: {rec['Mean_SSIM']:.4f}, VIFP: {rec['Mean_VIFP']:.4f}")

    # Save unified per-pair and summary
    df_per_pair = pd.DataFrame(all_per_pair)
    df_summary = pd.DataFrame(summary_records)

    df_per_pair.to_csv("results/dev_ablation_per_pair_unified.csv", index=False)
    df_summary.to_csv("results/dev_ablation_summary_unified.csv", index=False)

    # Compute ranking according to selection_rule.md
    df_ranked = df_summary.copy()
    df_ranked["Rank_QABF"] = df_ranked["Mean_QABF"].rank(ascending=False, method="min")
    df_ranked["Rank_NABF"] = df_ranked["Mean_NABF"].rank(ascending=True, method="min")
    df_ranked["Rank_MI"] = df_ranked["Mean_MI"].rank(ascending=False, method="min")
    df_ranked["Rank_SSIM"] = df_ranked["Mean_SSIM"].rank(ascending=False, method="min")
    df_ranked["Rank_VIFP"] = df_ranked["Mean_VIFP"].rank(ascending=False, method="min")

    df_ranked["Mean_Rank"] = (
        df_ranked["Rank_QABF"] + df_ranked["Rank_NABF"] + df_ranked["Rank_MI"] +
        df_ranked["Rank_SSIM"] + df_ranked["Rank_VIFP"]
    ) / 5.0

    df_ranked = df_ranked.sort_values(by=["Mean_Rank", "Mean_Latency_ms"], ascending=[True, True]).reset_index(drop=True)
    df_ranked["Overall_Rank"] = range(1, len(df_ranked) + 1)
    df_ranked.to_csv("results/dev_ablation_ranked_unified.csv", index=False)

    print("\n" + "="*80)
    print("UNIFIED DEV-SET ABLATION RANKING (selection_rule.md)")
    print("="*80)
    cols = ["Overall_Rank", "Config", "Mean_Rank", "Mean_QABF", "Mean_NABF", "Mean_MI", "Mean_SSIM", "Mean_VIFP", "Mean_Latency_ms"]
    print(df_ranked[cols].to_string(index=False))

if __name__ == "__main__":
    main()
