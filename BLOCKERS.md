# BLOCKERS: Dataset & Baseline Acquisition Audit

*Created pursuant to Non-Negotiable Hard Rule 2: If a dataset or pretrained baseline cannot be downloaded (no network, broken link, authentication barrier), do not substitute silently. Record exact status and manual download instructions.*

---

## 1. Deep Learning Baseline Pretrained Checkpoints Status

### BLOCKED-01: FusionGAN (Ma et al., Information Fusion 2019)
- **Status**: BLOCKED (Runtime / Framework Incompatibility)
- **Official Repository**: `https://github.com/jiayi-ma/FusionGAN`
- **Root Cause**: FusionGAN is officially authored in legacy **TensorFlow 1.x** (`tf.contrib`), which is strictly incompatible with modern Python 3.10+ environments (the current host environment runs Python 3.12). TensorFlow 1.x packages are only available up to Python 3.6.
- **Manual Reproduction Instructions**:
  1. Set up an isolated Python 3.6 or Docker environment: `docker pull tensorflow/tensorflow:1.15.5-gpu-py3`.
  2. Clone `https://github.com/jiayi-ma/FusionGAN.git`.
  3. Download the checkpoint files from the author's Baidu Netdisk link (`pan.baidu.com/s/1eS6iPiu` password: `dwbv`) or Google Drive.
  4. Place checkpoint files under `checkpoint/` and execute `python main.py`.

---

### BLOCKED-02: DDFM (Zhao et al., ICCV 2023 Oral)
- **Status**: BLOCKED (Large Checkpoint Host Download Barrier)
- **Official Repository**: `https://github.com/Zhaozixiang1228/MMIF-DDFM`
- **Root Cause**: DDFM relies on OpenAI's unconditional diffusion backbone `256x256_diffusion_uncond.pt` (~2.1 GB) and target checkpoints hosted on Google Drive / Baidu Netdisk. Automated downloading via headless CLI `curl`/`wget` fails because Google Drive returns an interactive HTML confirmation page ("Virus scan warning: File exceeds maximum scan size").
- **Manual Download Instructions**:
  1. Open browser and navigate to the Google Drive link in `https://github.com/Zhaozixiang1228/MMIF-DDFM/blob/main/README.md`.
  2. Click "Download anyway" to obtain `256x256_diffusion_uncond.pt`.
  3. Place the file at `baselines/deep/ddfm/checkpoints/256x256_diffusion_uncond.pt`.

---

## 2. Successfully Acquired & Verified Baselines

| Baseline | Type | Source / Model Origin | Execution Status |
|---|---|---|---|
| **Averaging** | Classical Baseline | Native Vectorized Implementation | **RUNNING** (Verified on 5 pairs) |
| **DWT** | Classical Wavelet | PyWavelets (`db2`, 2 levels) | **RUNNING** (Verified on 5 pairs) |
| **GFF** | Classical Guided Filter | Li et al., IEEE TIP 2013 | **RUNNING** (Verified on 5 pairs) |
| **ADF** | Classical Anisotropic Diffusion | Bavirisetti & Dhuli, Inf. Fusion 2016 | **RUNNING** (Verified in test suite) |
| **CBF** | Classical Cross Bilateral Filter | Kumar, IEEE SPL 2015 | **RUNNING** (Verified in test suite) |
| **NSCT (Substitute)** | Classical Non-Subsampled Transform | Stationary Wavelet Transform (`swt2`) + Directional Steerable Energy | **RUNNING** (Documented substitute for MATLAB MEX) |
| **DenseFuse** | Deep Learning (AE) | `hli1221/imagefusion_densefuse` | Model wrapper implemented; weights script configured |
| **U2Fusion** | Deep Learning (Unsupervised) | `hanna-xu/U2Fusion-pytorch` | PyTorch model + `final.pth` checkpoint |
| **SwinFusion** | Deep Learning (Transformer) | `Linfeng-Tang/SwinFusion` | PyTorch model + weights |
| **Proposed CADM-MSWN** | Proposed Training-Free Core | Native Vectorized Implementation | **RUNNING** (100% tests passed) |
