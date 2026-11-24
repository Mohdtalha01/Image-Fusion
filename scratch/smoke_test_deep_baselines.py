import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
import time
import os
import json

print("=== SMOKE TESTING ALL 5 DEEP LEARNING BASELINES ON DEV PAIR ===")

# Test pair: MSRS dev pair
ir_path = "data/raw/msrs_repo/train/ir/00003N.png"
vis_path = "data/raw/msrs_repo/train/vi/00003N.png"

assert os.path.exists(ir_path), f"Missing {ir_path}"
assert os.path.exists(vis_path), f"Missing {vis_path}"

im_ir = cv2.imread(ir_path)
im_vis = cv2.imread(vis_path)
print(f"Loaded source IR shape: {im_ir.shape}, VIS shape: {im_vis.shape}")

results = {}

# 1. DenseFuse
print("\n--- 1. Testing DenseFuse ---")
try:
    from baselines import fuse_densefuse
    t0 = time.perf_counter()
    fused_df = fuse_densefuse(im_ir, im_vis)
    dt = time.perf_counter() - t0
    assert fused_df is not None and fused_df.shape == im_vis.shape
    assert 0.0 <= fused_df.min() and fused_df.max() <= 1.0
    results["DenseFuse"] = {"status": "PASS", "latency_s": round(dt, 3), "shape": str(fused_df.shape)}
    print(f"DenseFuse PASS ({dt:.3f}s)")
except Exception as e:
    results["DenseFuse"] = {"status": "FAIL", "error": str(e)}
    print(f"DenseFuse FAIL: {e}")

# 2. U2Fusion
print("\n--- 2. Testing U2Fusion ---")
try:
    from baselines import fuse_u2fusion
    t0 = time.perf_counter()
    fused_u2 = fuse_u2fusion(im_ir, im_vis)
    dt = time.perf_counter() - t0
    assert fused_u2 is not None and fused_u2.shape == im_vis.shape
    assert 0.0 <= fused_u2.min() and fused_u2.max() <= 1.0
    results["U2Fusion"] = {"status": "PASS", "latency_s": round(dt, 3), "shape": str(fused_u2.shape)}
    print(f"U2Fusion PASS ({dt:.3f}s)")
except Exception as e:
    results["U2Fusion"] = {"status": "FAIL", "error": str(e)}
    print(f"U2Fusion FAIL: {e}")

# 3. SwinFusion
print("\n--- 3. Testing SwinFusion ---")
try:
    from baselines import fuse_swinfusion
    t0 = time.perf_counter()
    fused_swin = fuse_swinfusion(im_ir, im_vis, task="ivif")
    dt = time.perf_counter() - t0
    assert fused_swin is not None and fused_swin.shape == im_vis.shape
    assert 0.0 <= fused_swin.min() and fused_swin.max() <= 1.0
    results["SwinFusion"] = {"status": "PASS", "latency_s": round(dt, 3), "shape": str(fused_swin.shape)}
    print(f"SwinFusion PASS ({dt:.3f}s)")
except Exception as e:
    results["SwinFusion"] = {"status": "FAIL", "error": str(e)}
    print(f"SwinFusion FAIL: {e}")

# 4. CDDFuse
print("\n--- 4. Testing CDDFuse ---")
try:
    from baselines import fuse_cddfuse
    t0 = time.perf_counter()
    fused_cdd = fuse_cddfuse(im_ir, im_vis, task="ivif")
    dt = time.perf_counter() - t0
    assert fused_cdd is not None and fused_cdd.shape == im_vis.shape
    assert 0.0 <= fused_cdd.min() and fused_cdd.max() <= 1.0
    results["CDDFuse"] = {"status": "PASS", "latency_s": round(dt, 3), "shape": str(fused_cdd.shape)}
    print(f"CDDFuse PASS ({dt:.3f}s)")
except Exception as e:
    results["CDDFuse"] = {"status": "FAIL", "error": str(e)}
    print(f"CDDFuse FAIL: {e}")

# 5. PIAFusion
print("\n--- 5. Testing PIAFusion ---")
try:
    from baselines import fuse_piafusion
    t0 = time.perf_counter()
    fused_pia = fuse_piafusion(im_ir, im_vis, task="ivif")
    dt = time.perf_counter() - t0
    assert fused_pia is not None and fused_pia.shape == im_vis.shape
    assert 0.0 <= fused_pia.min() and fused_pia.max() <= 1.0
    results["PIAFusion"] = {"status": "PASS", "latency_s": round(dt, 3), "shape": str(fused_pia.shape)}
    print(f"PIAFusion PASS ({dt:.3f}s)")
except Exception as e:
    results["PIAFusion"] = {"status": "FAIL", "error": str(e)}
    print(f"PIAFusion FAIL: {e}")

print("\n=== SMOKE TEST SUMMARY ===")
print(json.dumps(results, indent=2))
with open("results/deep_baselines_smoke_test.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved smoke test results to results/deep_baselines_smoke_test.json")
