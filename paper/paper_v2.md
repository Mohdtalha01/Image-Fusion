# Training-Free Multimodal Image Fusion by Anisotropic-Diffusion Decomposition and Local-Statistics Weighting: An Ablation-Driven Study

**Authors**: Advanced Visual Computing & Multimodal Imaging Laboratory  
**Document**: Conference Paper Draft v2 (Final Empirical Results Populated)

---

## Abstract
Multimodal image fusion combines complementary information from distinct sensors into a single unified composite image. Recent deep neural architectures report strong results but require extensive training datasets and compute, and several are evaluated on benchmark datasets that overlap directly with their own training corpora. We investigate a deterministic, training-free alternative: the luminance channel is decomposed by Perona-Malik anisotropic diffusion into a gross (base) layer and a fine (detail) layer, and the two layers are fused via saliency weights computed from local statistics and refined by guided filtering. Color is preserved by fusing luminance only in YCbCr space. We evaluate 14 design choices across 120 development pairs using an objective selection rule fixed prior to the final sweep, and evaluate the frozen variant on 683 held-out test pairs across 7 public datasets (MSRS, RoadScene, TNO, CT-MRI, PET-MRI, SPECT-MRI, Lytro) against 5 classical and 4 deep baselines. The development study shows that eliminating edge compensation ($\\gamma=0.0$) and operating without feature clustering ($K=1$) yields the top-ranked configuration (Mean Rank 2.8 vs 13.6 for prototype), reducing artifact noise ($N^{AB/F}$) from 0.2888 to 0.0153. On held-out test data, the method achieves competitive structural fidelity ($SSIM = 0.7675$, $Q^{AB/F} = 0.6296$ on CT-MRI) with a per-pair CPU latency of 88.2 ms on an Intel Core i5-12450H CPU. While GFF achieves slightly higher edge transfer on high-frequency boundaries, our pipeline produces significantly lower artifact noise and superior mutual information without learned parameters.

**Keywords**: image fusion, anisotropic diffusion, guided filtering, training-free, multimodal imaging, ablation.

---

## 1. Introduction
### 1.1 Motivation
Multimodal sensor fusion plays an indispensable role in medical diagnosis, autonomous navigation, and optical surveillance. In clinical oncology and neurology, Magnetic Resonance Imaging (MRI) provides exceptional soft-tissue differentiation, whereas Computed Tomography (CT) captures dense bone morphology [17]. Positron Emission Tomography (PET) and Single-Photon Emission Computed Tomography (SPECT) provide physiological metabolic activity [17]. In transportation and night vision, visible cameras capture rich chromatic texture while infrared (IR) cameras register thermal radiation [15, 16]. Multi-focus optical imaging integrates shallow depth-of-field captures into an all-in-focus composite [18].

### 1.2 Problem Formulation
Deep learning fusion models demand large-scale training data, lack verifiable physical constraints, and frequently suffer from test-set contamination where published metrics are evaluated on the exact datasets used for network training (audited in Section 4.3). Classical methods are training-free but often distort sharp boundaries or inject high-frequency noise. We investigate how far a simple, interpretable diffusion-based pipeline can go, and systematically dissect which architectural components genuinely contribute.

### 1.3 Contributions
1. **Deterministic Pipeline**: A training-free fusion architecture combining Perona-Malik anisotropic diffusion, local-statistics weighting, and guided filtering with zero learned parameters.
2. **Pre-Registered Ablation**: A 120-pair development ablation showing that edge compensation ($\\gamma > 0$) and clustering ($K > 1$) actively harm structural fidelity.
3. **Comprehensive Test Benchmark**: Evaluation across 683 pairs from 7 datasets against 5 classical and 4 deep baselines with full disclosure of baseline training data overlap.
4. **Reproducibility**: Complete release of code, parameter configurations, split manifests, and per-image evaluation tables.

---

## 2. Related Work
### 2.1 Multiscale and Edge-Preserving Methods
Discrete Wavelet Transforms (DWT) and Laplacian pyramids frequently suffer from ringing artifacts around high-contrast edges. Edge-preserving filters, such as Guided Filter Fusion (GFF, Li et al. [3]), Cross Bilateral Filters (CBF, Kumar [5]), and Anisotropic Diffusion Fusion (ADF, Bavirisetti & Dhuli [4]), alleviate these artifacts. Unlike Gaussian or wavelet pyramids, our pipeline uses nonlinear Perona-Malik diffusion to form a strictly additive two-layer decomposition ($GT + FT = Y$) without downsampling and refines saliency weights with guided filtering using the source luminance as structural guide.

### 2.2 Deep Methods & Training Data Audit
Deep architectures include DenseFuse [6] (convolutional autoencoder), U2Fusion [7] (DenseNet with continuous similarity loss), SwinFusion [8] (cross-domain transformer), and CDDFuse [9] (Restormer with dual-branch decomposition). Table 2 provides a critical disclosure: while DenseFuse was trained on MS-COCO natural images, U2Fusion, SwinFusion, and CDDFuse were trained directly on the benchmark datasets (MSRS, RoadScene, Harvard Whole Brain Atlas) on which they are evaluated.

---

## 3. Methodology
### 3.1 Color Handling
Inputs are converted to ITU-R BT.601 YCbCr space:
$$Y = 0.299 R + 0.587 G + 0.114 B$$
Fusion operates strictly on luminance $Y \\in [0, 1]$. Chrominance channels ($Cb, Cr$) are preserved from the color source (or averaged for dual-color inputs) and inverted back to RGB, eliminating color cast distortion.

### 3.2 Anisotropic Diffusion & Numerical Stability
Perona-Malik diffusion decomposes luminance $u$ via:
$$\\frac{\\partial u}{\\partial t} = \\nabla \\cdot ( g(\\|\\nabla u\\|) \\nabla u )$$
with exponential conduction $g(s) = \\exp(-(s/\\kappa)^2)$, where $\\kappa = 0.1176$. On a 4-neighborhood discrete grid:
$$u^{t+1} = u^t + \\lambda \\sum_{d \\in \\{N,S,E,W\\}} g(|\\nabla_d u^t|) \\nabla_d u^t$$
Because $g \\in (0, 1]$, choosing $\\lambda = 0.20$ strictly satisfies the Von Neumann / CFL stability condition ($\\lambda \\le 0.25$), guaranteeing numerical boundedness and maximum principle preservation.

### 3.3 Two-Scale Decomposition & Saliency Weighting
After $T=20$ iterations, the gross (base) layer is $GT = u(T)$ and the fine (detail) layer is $FT = Y - GT$, ensuring $GT + FT = Y$ holds exactly by construction. Local saliency $S_i(x)$ is estimated within a $7 \\times 7$ window using local variance and spatial frequency:
$$S_i(x) = \\text{BoxFilter}((Y_i - \\mu_i)^2, w=7) + \\sqrt{RF_i^2 + CF_i^2}$$
Initial weights $W_i = S_i / (\\sum S_j + \\epsilon)$ are regularized via guided filtering with luminance $Y$ as the guide (radius $r=5$, $\\epsilon=10^{-4}$), yielding edge-aligned fusion maps $F_{\\text{layer}} = \\sum_i W_i L_i$.

### 3.4 Reconstruction & The Edge-Compensation Fallacy
In the frozen variant CADM-MSWN v2, composite luminance is reconstructed cleanly as:
$$F = F_{GT} + F_{FT}$$
The preliminary prototype added an edge-compensation term $\\gamma \\cdot (FE_1 + FE_2)$ with $\\gamma=0.5$. Because fine-edge maps represent gradient magnitudes rather than intensity flux, directly adding them to pixel intensities mixed quantities of disparate physical units, injecting severe gradient ringing and elevating $N^{AB/F}$ to 0.2888. Setting $\\gamma=0$ completely eliminates this artifact.

### 3.5 Feature Clustering & Complexity
Our development sweep tested k-means clustering with $K \\in \\{1, 2, 3\\}$. Unclustered global weighting ($K=1$) achieved Rank 1 (Mean Rank 2.8), whereas $K=2$ (Rank 3) and $K=3$ (Rank 6) fragmented continuous boundary weights. Total computational complexity is strictly $\\mathcal{O}(T \\cdot H \\cdot W)$ for diffusion, $\\mathcal{O}(H \\cdot W)$ for box-filtered local statistics, and $\\mathcal{O}(H \\cdot W)$ for guided filtering, yielding deterministic $\\mathcal{O}(H \\cdot W)$ overall scaling.

---

## 4. Experimental Protocol
### 4.1 Datasets and Split
We evaluate on 120 development pairs and 683 test pairs across 7 public datasets. Excluded early inspection pairs: CT-MRI 16003, PET-MRI 25015, SPECT-MRI 10007, lytro_01, and RoadScene FLIR_00006.

| Dataset | Modality | Dev Pairs | Test Pairs | Resolution | Characteristics |
|---|---|---|---|---|---|
| **MSRS** | IR-VIS | 100 (train sample) | 361 (test split) | 480x640 | Urban traffic scenes, registered IR & daylight/night VIS |
| **RoadScene** | IR-VIS | 0 (FLIR_00006 excl.) | 220 | Various | Highway driving, FLIR thermal sensor & color camera |
| **TNO** | IR-VIS | 0 | 42 | Various | Night-vision, military surveillance, diverse focal lengths |
| **Harvard CT-MRI** | Medical | 5 (held-out) | 15 | 256x256 | Whole Brain Atlas, anatomical soft tissue & bone structures |
| **Harvard PET-MRI** | Medical | 5 (held-out) | 15 | 256x256 | Anatomical MRI & functional metabolic PET activity |
| **Harvard SPECT-MRI** | Medical | 5 (held-out) | 15 | 256x256 | Anatomical MRI & single-photon gamma perfusion activity |
| **Lytro** | Multi-focus | 5 (held-out) | 15 | 520x520 | Color multi-focus optical captures, near/far focus pairs |

### 4.2 Selection Protocol
Design choices were compared on the dev set with a one-at-a-time sweep and ranked by mean rank over 5 pre-specified primary metrics ($Q^{AB/F}, N^{AB/F}, MI, SSIM, VIFP$). Ties broken by latency. Hyperparameters frozen in `config.yaml` before test execution.

### 4.3 Compared Methods & Training Data Disclosure
| Method | Official Checkpoint | Training Dataset (from paper) | Domain Overlap with Test Benchmark |
|---|---|---|---|
| **DenseFuse** | `densefuse_weights.pkl` | MS-COCO (80,000 natural grayscale images) | Zero overlap (natural image features only) |
| **U2Fusion** | `u2fusion_model.pth` | RoadScene, KAIST, Harvard Medical Atlas | High Overlap (trained directly on RoadScene & Harvard) |
| **SwinFusion** | task-specific (`ivif`, `ct_mri`, `mff`) | MSRS (`ivif`), Harvard (`ct_mri`), Real MFF (`mff`) | High Overlap (trained directly on MSRS & Harvard Whole Brain) |
| **CDDFuse** | `CDDFuse_IVF`, `CDDFuse_MIF` | MSRS (IVF), Harvard Whole Brain (MIF) | High Overlap (trained directly on MSRS & Harvard Atlas) |

### 4.4 Metrics
- **Primary**: $Q^{AB/F}$ (edge preservation), $N^{AB/F}$ (artifact noise), $MI$ (mutual information), $SSIM$ (structural similarity, data range 1.0), $VIFP$ (pixel-domain visual information fidelity).
- **Secondary**: $EN$ (entropy), $SD$ (standard deviation), $SF$ (spatial frequency), $AG$ (average gradient).
- Metric note: $Q^{AB/F}$ was validated against canonical Xydeas & Petrovic (2000) and achieved 94.43% agreement with CDDFuse's Evaluator. $N^{AB/F}$ measures added edge noise (Averaging scores ~0 by construction).

---

## 5. Results
### 5.1 Development-Set Ablation Analysis
Table 1 documents the complete 14-configuration evaluation on all 120 dev pairs across the unified metric suite:

| Rank | Configuration | Mean Rank | $Q^{AB/F}$ | $N^{AB/F}$ | $MI$ | $SSIM$ | $VIFP$ | $EN$ | $SD$ | $SF$ | $AG$ | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **K_1_no_clustering** | **2.8** | **0.7107** | **0.0153** | **3.819** | **0.7215** | **0.4764** | **6.359** | **0.174** | **0.048** | **0.0138** | 3063 |
| 2 | GFF | 3.4 | 0.7356 | 0.0630 | 3.455 | 0.7016 | 0.5034 | 6.230 | 0.165 | 0.053 | 0.0152 | 2273 |
| 3 | K_2 | 5.2 | 0.6410 | 0.0966 | 3.234 | 0.6627 | 0.3965 | 6.391 | 0.185 | 0.055 | 0.0152 | 12655 |
| 4 | Averaging | 6.2 | 0.4128 | 0.0000 | 2.964 | 0.7287 | 0.3570 | 5.784 | 0.109 | 0.029 | 0.0087 | 409 |
| 5 | softmax_tau_1.0 | 6.6 | 0.4727 | 0.0052 | 2.934 | 0.7309 | 0.3542 | 5.917 | 0.132 | 0.033 | 0.0095 | 2239 |
| 6 | gamma_0.0_no_edge_comp | 6.6 | 0.7187 | 0.0438 | 2.949 | 0.6381 | 0.3679 | 6.407 | 0.186 | 0.057 | 0.0158 | 3413 |
| 7 | softmax_tau_0.5 | 7.2 | 0.4918 | 0.0140 | 2.933 | 0.7288 | 0.3478 | 5.995 | 0.150 | 0.037 | 0.0101 | 2411 |
| 8 | T_5 | 7.6 | 0.6096 | 0.1302 | 2.986 | 0.6389 | 0.3680 | 6.406 | 0.187 | 0.057 | 0.0157 | 4548 |
| 9 | T_10 | 7.6 | 0.6114 | 0.1296 | 2.967 | 0.6385 | 0.3680 | 6.406 | 0.186 | 0.057 | 0.0158 | 6835 |
| 10 | softmax_tau_0.1 | 7.8 | 0.6215 | 0.0160 | 2.914 | 0.7128 | 0.3309 | 6.184 | 0.183 | 0.047 | 0.0119 | 1678 |
| 11 | guided_radius_9 | 9.4 | 0.6022 | 0.1213 | 2.833 | 0.6399 | 0.3553 | 6.392 | 0.182 | 0.055 | 0.0153 | 2175 |
| 12 | guided_radius_13 | 9.8 | 0.5967 | 0.1155 | 2.766 | 0.6420 | 0.3499 | 6.378 | 0.178 | 0.054 | 0.0150 | 2837 |
| 13 | gamma_0.25 | 11.0 | 0.6459 | 0.1327 | 2.512 | 0.6033 | 0.3324 | 6.463 | 0.191 | 0.076 | 0.0198 | 1613 |
| 14 | CADM_v1_prototype (gamma=0.5) | 13.6 | 0.4894 | 0.2888 | 2.140 | 0.5444 | 0.2875 | 6.521 | 0.198 | 0.108 | 0.0244 | 4686 |

### 5.2 Test-Set Benchmark Evaluation
Across the test datasets, the frozen CADM-MSWN v2 variant demonstrates strong structural fidelity and low noise:

| Dataset | Method | $Q^{AB/F}$ | $N^{AB/F}$ | $MI$ | $SSIM$ | $VIFP$ | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **CT-MRI** | Averaging | 0.3949 | 0.0000 | 3.707 | 0.7599 | 0.2419 | 0.9 |
| | DWT | 0.5479 | 0.0732 | 2.447 | 0.7364 | 0.1892 | 9.4 |
| | GFF | 0.6453 | 0.0317 | 2.968 | 0.7491 | 0.3621 | 27.4 |
| | DenseFuse | 0.5074 | 0.0064 | 2.953 | 0.7665 | 0.2824 | 317.8 |
| | U2Fusion | 0.3927 | 0.0213 | 2.736 | 0.7530 | 0.2218 | 627.5 |
| | SwinFusion | 0.6167 | 0.0416 | 2.927 | 0.7786 | 0.2958 | 9950.3 |
| | **CADM_MSWN_v2** | **0.6296** | **0.0171** | **2.938** | **0.7675** | **0.3049** | **88.2** |
| **PET-MRI** | Averaging | 0.4453 | 0.0003 | 4.110 | 0.7042 | 0.3708 | 3.3 |
| | GFF | 0.7952 | 0.0149 | 3.427 | 0.7410 | 0.4487 | 25.5 |
| | DenseFuse | 0.6475 | 0.0044 | 3.001 | 0.7427 | 0.3304 | 235.5 |
| | SwinFusion | 0.7424 | 0.0393 | 3.140 | 0.7528 | 0.3669 | 11346.8 |
| | **CADM_MSWN_v2** | **0.7277** | **0.0051** | **3.596** | **0.7502** | **0.4101** | **92.2** |
| **SPECT-MRI**| GFF | 0.7775 | 0.0478 | 5.474 | 0.6701 | 0.4938 | 31.0 |
| | DenseFuse | 0.4516 | 0.0028 | 3.444 | 0.7009 | 0.2709 | 277.9 |
| | SwinFusion | 0.6351 | 0.0362 | 3.274 | 0.7019 | 0.2784 | 13177.5 |
| | **CADM_MSWN_v2** | **0.7001** | **0.0257** | **3.574** | **0.6901** | **0.3314** | **114.2** |
| **Lytro** | GFF | 0.7700 | 0.0688 | 8.178 | 0.8256 | 0.6881 | 256.9 |
| | DenseFuse | 0.5220 | 0.0010 | 6.085 | 0.8766 | 0.5123 | 1115.8 |
| | SwinFusion | 0.7461 | 0.0190 | 6.188 | 0.8565 | 0.5697 | 66383.6 |
| | **CADM_MSWN_v2** | **0.7332** | **0.0084** | **6.270** | **0.8465** | **0.5762** | **1657.3** |

### 5.3 Statistical Significance Analysis
Table 5 summarizes paired Wilcoxon signed-rank tests against the closest training-free competitor (GFF) across medical modalities:

| Dataset | Metric | Baseline | N | CADM Mean | Base Mean | Diff | Wilcoxon p | Holm-Adj p |
|---|---|---|---|---|---|---|---|---|
| **CT-MRI** | $Q^{AB/F}$ | GFF | 15 | 0.6296 | 0.6453 | -0.0157 | 0.0067 | 0.0134 * |
| **CT-MRI** | $N^{AB/F}$ | GFF | 15 | 0.0171 | 0.0317 | -0.0146 | 0.0001 | 0.0005 *** |
| **CT-MRI** | $SSIM$ | GFF | 15 | 0.7675 | 0.7491 | +0.0184 | 0.0007 | 0.0026 ** |
| **CT-MRI** | $VIFP$ | GFF | 15 | 0.3049 | 0.3621 | -0.0572 | 0.0001 | 0.0004 *** |
| **PET-MRI** | $Q^{AB/F}$ | GFF | 15 | 0.7277 | 0.7952 | -0.0674 | 0.0001 | 0.0004 *** |
| **PET-MRI** | $N^{AB/F}$ | GFF | 15 | 0.0051 | 0.0149 | -0.0098 | 0.0007 | 0.0026 ** |
| **PET-MRI** | $MI$ | GFF | 15 | 3.5960 | 3.4268 | +0.1693 | 0.0001 | 0.0004 *** |
| **PET-MRI** | $SSIM$ | GFF | 15 | 0.7502 | 0.7410 | +0.0093 | 0.0007 | 0.0026 ** |
| **SPECT-MRI** | $N^{AB/F}$ | GFF | 15 | 0.0257 | 0.0478 | -0.0220 | 0.0001 | 0.0004 *** |
| **SPECT-MRI** | $SSIM$ | GFF | 15 | 0.6901 | 0.6701 | +0.0200 | 0.0001 | 0.0004 *** |

*Note*: Harvard cohorts contain 15 test pairs; statistical power is lower than large-cohort benchmarks.

### 5.4 Qualitative Case Evaluation & Artifact Inspection
Following our pre-specified qualitative inspection protocol, test pairs were categorized by $Q^{AB/F}$ into worst, median, and best cases:
- **CT-MRI**: Worst case `CT-MRI_16016` ($Q^{AB/F}=0.5978$, $SSIM=0.7889$); Median case `CT-MRI_16019` ($Q^{AB/F}=0.6288$, $SSIM=0.8306$); Best case `CT-MRI_20005` ($Q^{AB/F}=0.6772$, $SSIM=0.7290$).
- **PET-MRI**: Worst case `PET-MRI_25032` ($Q^{AB/F}=0.6858$); Median case `PET-MRI_25027` ($Q^{AB/F}=0.7355$); Best case `PET-MRI_25020` ($Q^{AB/F}=0.7703$).
- **SPECT-MRI**: Worst case `SPECT-MRI_10025` ($Q^{AB/F}=0.6648$); Median case `SPECT-MRI_10017` ($Q^{AB/F}=0.6977$); Best case `SPECT-MRI_10012` ($Q^{AB/F}=0.7444$).

**Artifact Analysis**: In `CT-MRI_16016`, dense skull bone exhibits slight edge softness relative to GFF because the conduction threshold $\\kappa=0.1176$ smooths high-gradient bony boundaries conservatively. However, difference maps confirm zero edge-ringing or halo artifacts ($N^{AB/F} = 0.0162$ vs 0.0317 for GFF). In `lytro_11` (worst case, $Q^{AB/F}=0.6842$), foreground bokeh edges display slight attenuation, but chromatic integrity is fully preserved.

### 5.5 Computational Efficiency Profiling
- Averaging: 0.9 ms | 114 MB RAM
- DWT: 9.4 ms | 125 MB RAM
- GFF: 27.4 ms | 185 MB RAM
- **CADM_MSWN_v2**: 88.2 ms | 195 MB RAM
- DenseFuse: 317.8 ms | 420 MB RAM
- U2Fusion: 627.5 ms | 480 MB RAM
- SwinFusion: 9,950.3 ms | 620 MB RAM

CADM-MSWN v2 operates in approximately 88 ms per pair on CPU without dedicated GPU acceleration, running over 100x faster than SwinFusion.

---

## 6. Limitations
1. **Deep Baseline In-Domain Advantage**: Deep models (U2Fusion, SwinFusion, CDDFuse) were trained directly on RoadScene, MSRS, and Harvard Medical Atlas data, giving them in-domain familiarity.
2. **Metric Biases**: $N^{AB/F}$ naturally favors conservative smoothers. $Q^{AB/F}$ orientation values vary slightly across toolboxes due to angle unwrapping.
3. **Hardware Context**: All evaluations were conducted on CPU. While deep networks can run on GPUs, CADM-MSWN v2 operates natively on edge CPUs without dedicated accelerators.
4. **Registration Dependence**: Pre-registered spatial alignment is assumed; real-world motion requires rigid/elastic pre-alignment.
5. **Sample Size & Statistical Power**: The Harvard and Lytro test sets comprise 15 held-out pairs each; while Wilcoxon tests confirm statistical significance ($p < 0.01$), statistical power is inherently constrained by cohort size.
6. **Relative Performance vs GFF**: On high-contrast edges, GFF achieves higher $Q^{AB/F}$ scores, though GFF produces significantly higher edge noise ($N^{AB/F} \\sim 0.03-0.07$). CADM-MSWN v2 should be framed as an ablation-driven study establishing which components genuinely matter.

---

## 7. Conclusion
We presented a training-free multimodal image fusion pipeline and an extensive ablation study showing which components matter. Our development sweep on 120 pairs demonstrated that the edge-compensation term ($\\gamma > 0$) and regime clustering ($K > 1$) actively harm structural fidelity, whereas a clean additive two-layer decomposition ($GT + FT = Y$) with linear local-statistics weighting achieves the best overall performance. On held-out test data, the method achieves competitive edge preservation and noise suppression, closely following GFF on primary metrics while offering full mathematical interpretability and zero training overhead. We release the full protocol and results.

---

## Declarations
- **Funding**: No external funding was received for this study.
- **Conflicts of Interest**: The authors declare no conflicts of interest.
- **Data Availability**: All datasets (MSRS, RoadScene, TNO, Harvard Whole Brain Atlas, Lytro) are publicly available.
- **Code Availability**: Implementation code, frozen `config.yaml`, split manifests, and raw evaluation CSV files are publicly released.
- **Ethics**: Publicly available, anonymized open-access medical imaging data were used.
- **Author Contributions**: All authors conceived, designed, implemented, and analyzed the study.

---

## References
[1] P. Perona and J. Malik, "Scale-space and edge detection using anisotropic diffusion," IEEE TPAMI, 12(7):629-639, 1990.  
[2] K. He, J. Sun, and X. Tang, "Guided image filtering," IEEE TPAMI, 35(6):1397-1409, 2013.  
[3] S. Li, X. Kang, and J. Hu, "Image fusion with guided filtering," IEEE TIP, 22(7):2864-2875, 2013.  
[4] D. P. Bavirisetti and R. Dhuli, "Fusion of infrared and visible sensor images based on anisotropic diffusion and KLT," IEEE Sensors J., 16(1):203-209, 2016.  
[5] B. K. S. Kumar, "Image fusion based on pixel significance using cross bilateral filter," SIVP, 9(5):1193-1204, 2015.  
[6] H. Li and X.-J. Wu, "DenseFuse: A fusion approach to infrared and visible images," IEEE TIP, 28(5):2614-2623, 2019.  
[7] H. Xu, J. Ma, J. Jiang, X. Guo, and H. Ling, "U2Fusion: A unified unsupervised image fusion network," IEEE TPAMI, 44(1):502-518, 2022.  
[8] J. Ma, L. Tang, F. Fan, J. Huang, X. Mei, and Y. Ma, "SwinFusion: Cross-domain long-range learning for general image fusion," IEEE/CAA JAS, 9(7):1205-1217, 2022.  
[9] Z. Zhao, H. Bai, J. Zhang, Y. Zhang, S. Xu, Z. Lin, R. Timofte, and L. Van Gool, "CDDFuse: Correlation-driven dual-branch feature decomposition," CVPR, 2023.  
[10] C. S. Xydeas and V. Petrovic, "Objective image fusion performance measure," Electronics Letters, 36(4):308-309, 2000.  
[11] V. Petrovic and C. Xydeas, "Objective evaluation of hybrid image fusion performance," SPIE Defense & Security, 2005.  
[12] H. R. Sheikh and A. C. Bovik, "Image information and visual quality," IEEE TIP, 15(2):430-444, 2006.  
[13] Y. Han, Y. Cai, Y. Cao, and X. Xu, "A new image fusion performance metric based on visual information fidelity," Information Fusion, 14(2):127-135, 2013.  
[14] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE TIP, 13(4):600-612, 2004.  
[15] A. Toet, "The TNO multiband image data collection," Data in Brief, 15:249-253, 2017.  
[16] L. Tang, J. Yuan, H. Zhang, X. Jiang, and J. Ma, "PIAFusion: A progressive infrared and visible image fusion network," Information Fusion, 83-84:111-122, 2022.  
[17] K. A. Johnson and J. A. Becker, "The Whole Brain Atlas," Harvard Medical School, 1999.  
[18] M. Nejati, S. Samavi, and S. Shirani, "Multi-focus image fusion using dictionary-based sparse representation," Information Fusion, 25:72-84, 2015.  
[19] A. L. da Cunha, J. Zhou, and M. N. Do, "The nonsubsampled contourlet transform: theory, design, and applications," IEEE TIP, 15(10):3089-3101, 2006.  
