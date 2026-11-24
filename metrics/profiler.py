"""Computational profiler for image fusion algorithms.

Measures:
    - Wall-clock latency (CPU / GPU) with warmup runs and averaged repeats
    - Peak Process RSS RAM consumption via psutil (CPU)
    - Peak GPU VRAM consumption via torch.cuda.max_memory_allocated (GPU)
    - Learnable parameter count (0 for training-free methods)
    - Hardware environment logging (exact CPU and GPU model)
"""

import time
import os
import platform
import subprocess
from typing import Callable, Any, Dict, Tuple, Optional
import numpy as np
import psutil
import torch


def get_hardware_info() -> Dict[str, Any]:
    """Detect and return exact CPU and GPU hardware specifications."""
    cpu_name = platform.processor() or "Unknown CPU"

    # Try winreg on Windows for exact marketing name
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            cpu_name = cpu_name.strip()
        except Exception:
            pass

    # Detect GPU model
    gpu_name = "None"
    gpu_memory_total_mb = 0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    else:
        # Check nvidia-smi if on machine with GPU but running PyTorch CPU build
        try:
            out = subprocess.check_output(
                "nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader",
                shell=True,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            if out:
                parts = [p.strip() for p in out.split(",")]
                gpu_name = parts[0]
                if len(parts) > 1:
                    gpu_memory_total_mb = parts[1]
        except Exception:
            pass

    return {
        "cpu_model": cpu_name,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "gpu_model": gpu_name,
        "gpu_memory_total": gpu_memory_total_mb,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def count_parameters(model: Any = None) -> int:
    """Return learnable parameter count of model. Returns 0 for training-free methods."""
    if model is None:
        return 0

    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))

    return 0


def profile_execution(
    fuse_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    img1: np.ndarray,
    img2: np.ndarray,
    warmup_runs: int = 3,
    measured_runs: int = 5,
    model: Any = None,
) -> Dict[str, Any]:
    """Profile latency, process RSS peak, GPU VRAM peak, and parameter count.

    Args:
        fuse_fn: Callable taking (img1, img2) and returning fused image.
        img1: First test image.
        img2: Second test image.
        warmup_runs: Number of discarded initial executions.
        measured_runs: Number of timed executions.
        model: Optional model object for parameter counting.

    Returns:
        Dictionary of profiling metrics:
            - mean_latency_ms: Average time in milliseconds
            - std_latency_ms: Standard deviation of time
            - peak_ram_rss_mb: Peak resident set size (RSS) in MB
            - peak_vram_mb: Peak GPU allocated memory via torch.cuda in MB
            - parameter_count: Learnable parameters (0 for training-free)
            - hardware: Exact hardware and OS metadata
    """
    proc = psutil.Process(os.getpid())

    # Warmup runs
    for _ in range(warmup_runs):
        _ = fuse_fn(img1, img2)

    # Reset GPU stats if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Latency timing and memory sampling
    timings = []
    peak_rss = proc.memory_info().rss

    for _ in range(measured_runs):
        t0 = time.perf_counter()
        _ = fuse_fn(img1, img2)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

        curr_rss = proc.memory_info().rss
        if curr_rss > peak_rss:
            peak_rss = curr_rss

    # Peak VRAM
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = round(float(torch.cuda.max_memory_allocated() / (1024 * 1024)), 2)

    params = count_parameters(model)
    hardware = get_hardware_info()

    return {
        "mean_latency_ms": round(float(np.mean(timings)), 2),
        "std_latency_ms": round(float(np.std(timings)), 2),
        "peak_ram_rss_mb": round(float(peak_rss / (1024 * 1024)), 2),
        "peak_vram_mb": peak_vram_mb,
        "parameter_count": params,
        "hardware": hardware,
    }
