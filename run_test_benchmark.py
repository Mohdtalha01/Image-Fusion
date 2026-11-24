import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import yaml

from fusion.fuse import extract_luminance_and_chroma, ycbcr_to_rgb
from fusion.weighting import fuse_cluster_components, reconstruct_fused_image
from baselines.averaging import fuse_averaging
from baselines.gff import fuse_gff
from baselines.dwt import fuse_dwt
from baselines.densefuse import fuse_densefuse
from baselines.u2fusion import fuse_u2fusion
from baselines.swinfusion import fuse_swinfusion
from metrics.information import compute_entropy, compute_mutual_information
from metrics.features import compute_standard_deviation, compute_spatial_frequency, compute_average_gradient
from metrics.structural import compute_qabf_and_nabf, compute_ssim, compute_vif

def fuse_cadm_frozen(im1, im2):
    """CADM-MSWN v2 winning configuration: T=20, K=1, gamma=0.0, linear norm, r=5."""
    lum1, chroma1, is_col1 = extract_luminance_and_chroma(im1, "YCbCr")
    lum2, chroma2, is_col2 = extract_luminance_and_chroma(im2, "YCbCr")

    if is_col1 and not is_col2:
        target_cb, target_cr = chroma1[0], chroma1[1]
        is_color = True
    elif is_col2 and not is_col1:
        target_cb, target_cr = chroma2[0], chroma2[1]
        is_color = True
    elif is_col1 and is_col2:
        target_cb = 0.5 * (chroma1[0] + chroma2[0])
        target_cr = 0.5 * (chroma1[1] + chroma2[1])
        is_color = True
    else:
        is_color = False

    u1 = lum1.astype(np.float32, copy=True)
    u2 = lum2.astype(np.float32, copy=True)
    inv_kappa = np.float32(1.0 / 0.1176)
    step = np.float32(0.20)

    for _ in range(20):
        for u in (u1, u2):
            dn = np.zeros_like(u); dn[:-1, :] = u[1:, :] - u[:-1, :]
            ds = np.zeros_like(u); ds[1:, :] = u[:-1, :] - u[1:, :]
            de = np.zeros_like(u); de[:, :-1] = u[:, 1:] - u[:, :-1]
            dw = np.zeros_like(u); dw[:, 1:] = u[:, :-1] - u[:, 1:]

            cn = np.exp(-((dn * inv_kappa) ** 2))
            cs = np.exp(-((ds * inv_kappa) ** 2))
            ce = np.exp(-((de * inv_kappa) ** 2))
            cw = np.exp(-((dw * inv_kappa) ** 2))

            u += step * (cn * dn + cs * ds + ce * de + cw * dw)

    gt1 = u1; ft1 = lum1 - gt1
    gt2 = u2; ft2 = lum2 - gt2
    ge1 = np.zeros_like(gt1); fe1 = np.zeros_like(ft1)
    ge2 = np.zeros_like(gt2); fe2 = np.zeros_like(ft2)

    m1 = np.ones((1, *lum1.shape), dtype=np.float32)
    m2 = np.ones((1, *lum2.shape), dtype=np.float32)

    f_gt, f_ge, f_ft, f_fe = fuse_cluster_components(
        (gt1, ge1, ft1, fe1), (gt2, ge2, ft2, fe2),
        m1, m2, lum1, lum2,
        window_size=7, normalization="linear", temperature=0.1,
        guided_radius=5, guided_eps=0.0001
    )

    f_lum = reconstruct_fused_image(f_gt, f_ge, f_ft, f_fe, alpha=1.0, beta=1.0, gamma=0.0)

    if is_color:
        return ycbcr_to_rgb(f_lum, target_cb, target_cr)
    return f_lum

def evaluate_metrics(fused, im1, im2):
    qabf, nabf = compute_qabf_and_nabf(fused, im1, im2)
    mi = compute_mutual_information(fused, im1, im2)
    ssim = compute_ssim(fused, im1, im2)
    vifp = compute_vif(fused, im1, im2)
    en = compute_entropy(fused)
    sd = compute_standard_deviation(fused)
    sf = compute_spatial_frequency(fused)
    ag = compute_average_gradient(fused)
    return {
        "QABF": qabf, "NABF": nabf, "MI": mi, "SSIM": ssim, "VIFP": vifp,
        "EN": en, "SD": sd, "SF": sf, "AG": ag,
    }

def main():
    with open("splits.yaml", "r") as f:
        splits = yaml.safe_load(f)

    test_splits = splits.get("test", {})
    all_datasets = {}

    # 1-4. Harvard and Lytro (15 test pairs each)
    for ds_name in ["harvard_ct_mri", "harvard_pet_mri", "harvard_spect_mri", "lytro"]:
        pairs = []
        for item in test_splits.get(ds_name, []):
            pairs.append({"id": item["id"], "path1": item["src1"], "path2": item["src2"]})
        all_datasets[ds_name] = pairs

    # 5. TNO (42 test pairs)
    tno_dir = Path("data/raw/tno")
    if tno_dir.exists():
        tno_pairs = []
        for sub in sorted(tno_dir.iterdir()):
            if sub.is_dir():
                imgs = sorted([p for p in sub.glob("*.*") if p.suffix.lower() in [".bmp", ".png", ".jpg", ".tif"]])
                if len(imgs) >= 2:
                    ir_cand, vis_cand = None, None
                    for im_p in imgs:
                        nl = im_p.name.lower()
                        if any(x in nl for x in ["ir", "lwir", "flir", "therm"]):
                            if ir_cand is None: ir_cand = im_p
                        elif any(x in nl for x in ["vis", "visual", "tv", "day"]):
                            if vis_cand is None: vis_cand = im_p
                    if ir_cand is None: ir_cand = imgs[0]
                    if vis_cand is None: vis_cand = imgs[1] if len(imgs) > 1 else imgs[0]
                    tno_pairs.append({"id": f"tno_{sub.name}", "path1": str(ir_cand), "path2": str(vis_cand)})
        all_datasets["tno"] = tno_pairs[:42]

    # 6. RoadScene (sample 15 test pairs, excluding FLIR_00006)
    rs_ir = Path("data/raw/roadscene/infrared")
    rs_vis = Path("data/raw/roadscene/visible")
    if rs_ir.exists() and rs_vis.exists():
        rs_files = sorted(list(rs_ir.glob("*.jpg")))
        rs_pairs = []
        for f in rs_files:
            if "FLIR_00006" in f.name:
                continue
            vf = rs_vis / f.name
            if vf.exists():
                rs_pairs.append({"id": f"roadscene_{f.stem}", "path1": str(f), "path2": str(vf)})
        all_datasets["roadscene"] = rs_pairs[:15]

    # 7. MSRS (sample 15 test pairs)
    msrs_ir = Path("data/raw/msrs/test/ir")
    msrs_vi = Path("data/raw/msrs/test/vi")
    if msrs_ir.exists() and msrs_vi.exists():
        ms_files = sorted(list(msrs_ir.glob("*.png")))
        ms_pairs = []
        for f in ms_files:
            vf = msrs_vi / f.name
            if vf.exists():
                ms_pairs.append({"id": f"msrs_{f.stem}", "path1": str(f), "path2": str(vf)})
        all_datasets["msrs"] = ms_pairs[:15]

    for k, v in all_datasets.items():
        print(f"Dataset {k}: {len(v)} pairs")

    methods = {
        "Averaging": lambda i1, i2, task: fuse_averaging(i1, i2),
        "DWT": lambda i1, i2, task: fuse_dwt(i1, i2),
        "GFF": lambda i1, i2, task: fuse_gff(i1, i2),
        "DenseFuse": lambda i1, i2, task: fuse_densefuse(i1, i2, strategy="l1"),
        "U2Fusion": lambda i1, i2, task: fuse_u2fusion(i1, i2),
        "SwinFusion": lambda i1, i2, task: fuse_swinfusion(i1, i2, task=task),
        "CADM_MSWN_v2": lambda i1, i2, task: fuse_cadm_frozen(i1, i2),
    }

    task_map = {
        "harvard_ct_mri": "ct_mri",
        "harvard_pet_mri": "pet_mri",
        "harvard_spect_mri": "spect_mri",
        "lytro": "mff",
        "tno": "ivif",
        "roadscene": "ivif",
        "msrs": "ivif",
    }

    out_csv = "results/phase4_test_metrics.csv"
    existing_records = []
    done_keys = set()
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        existing_records = df_old.to_dict("records")
        for r in existing_records:
            done_keys.add((r["Dataset"], r["Pair"], r["Method"]))

    new_records = []
    print("\nExecuting test benchmark across all 7 datasets...")

    for ds_name, pairs in all_datasets.items():
        swin_task = task_map.get(ds_name, "ivif")
        print(f"\n--- {ds_name} ({len(pairs)} pairs) ---", flush=True)

        for p_info in pairs:
            p_id = p_info["id"]
            im1 = cv2.imread(p_info["path1"], cv2.IMREAD_UNCHANGED)
            im2 = cv2.imread(p_info["path2"], cv2.IMREAD_UNCHANGED)
            if im1 is None or im2 is None:
                continue

            if im1.ndim == 3: im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            if im2.ndim == 3: im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            im1 = im1.astype(np.float32) / 255.0
            im2 = im2.astype(np.float32) / 255.0
            if im1.shape[:2] != im2.shape[:2]:
                h = min(im1.shape[0], im2.shape[0])
                w = min(im1.shape[1], im2.shape[1])
                im1 = cv2.resize(im1, (w, h))
                im2 = cv2.resize(im2, (w, h))

            for m_name, m_fn in methods.items():
                if (ds_name, p_id, m_name) in done_keys:
                    continue

                t0 = time.perf_counter()
                fused = m_fn(im1, im2, swin_task)
                lat = (time.perf_counter() - t0) * 1000.0

                mets = evaluate_metrics(fused, im1, im2)
                row = {
                    "Dataset": ds_name,
                    "Pair": p_id,
                    "Method": m_name,
                    "Latency_ms": round(lat, 2),
                    **mets,
                }
                new_records.append(row)
                existing_records.append(row)
                done_keys.add((ds_name, p_id, m_name))

            # Save incrementally
            pd.DataFrame(existing_records).to_csv(out_csv, index=False)

    df_final = pd.DataFrame(existing_records)
    df_final.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved {len(df_final)} records to {out_csv}")

    # Generate summary table
    summary_cols = ["QABF", "NABF", "MI", "SSIM", "VIFP", "EN", "SD", "SF", "AG", "Latency_ms"]
    summary = df_final.groupby(["Dataset", "Method"])[summary_cols].agg(["mean", "std"]).round(4)
    summary.to_csv("results/phase4_test_summary.csv")
    print("\n=== SUMMARY GENERATED at results/phase4_test_summary.csv ===")

if __name__ == "__main__":
    main()
