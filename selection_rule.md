# CADM-MSWN v2 Pre-Registered Selection Rule and Experimental Protocol

**Timestamp**: 2026-11-24T13:03:00+05:30  
**Git Commit Requirement**: Pre-registered and committed prior to dev-set result inspection and Phase 4 test execution.  
**Evaluation Target**: 120 Development Pairs (`splits.yaml`):  
- 100 MSRS pairs (`data/MSRS/train`, seed 0)  
- 5 Harvard CT-MRI held-out pairs (`data/Harvard/CT-MRI/dev`)  
- 5 Harvard PET-MRI held-out pairs (`data/Harvard/PET-MRI/dev`)  
- 5 Harvard SPECT-MRI held-out pairs (`data/Harvard/SPECT-MRI/dev`)  
- 5 Lytro held-out pairs (`data/Lytro/dev`)  

---

## 1. Objective Decision Rule for Selecting CADM-MSWN v2

A candidate variant will be selected as **CADM-MSWN v2** if and only if it strictly outperforms the frozen baseline prototype according to the following objective ranking criterion:

### 1.1 The 5 Primary Metrics
Evaluation is governed strictly by the 5 primary decision metrics pre-registered in `LIMITATIONS.md`:
1. **$Q^{AB/F}$**: Edge information transfer ($\in [0, 1]$, **Higher is Better**).
2. **$N^{AB/F}$**: Fusion artifacts and excess noise ($\in [0, 1]$, **Lower is Better**).
3. **$MI$**: Mutual Information transferred from both source images (bits, **Higher is Better**).
4. **$SSIM$**: Mean Structural Similarity Index ($\in [0, 1]$, **Higher is Better**).
5. **$VIFF / VIFP$**: Visual Information Fidelity (**Higher is Better**; both pixel-domain VIFP and multiscale fusion VIFF evaluated).

### 1.2 Mathematical Formulation of Ranking Rule
Let $\mathcal{C} = \{c_1, c_2, \dots, c_M\}$ denote the set of candidate configurations (including the baseline prototype $c_{\text{base}}$).  
For each metric $m \in \{Q^{AB/F}, -N^{AB/F}, MI, SSIM, VIFF\}$, each configuration $c \in \mathcal{C}$ receives a rank $\text{Rank}_m(c) \in \{1, \dots, M\}$ based on its 120-pair mean metric value:
- For $Q^{AB/F}, MI, SSIM, VIFF, VIFP$: Highest mean value receives Rank 1 (best).
- For $N^{AB/F}$: **Lowest mean value receives Rank 1 (best)** (higher-is-worse correctly handled via ascending rank).

The **Mean Rank** of candidate $c$ across the primary metric set $\mathcal{M}_{\text{primary}}$ is defined as:
$$\bar{R}(c) = \frac{1}{|\mathcal{M}_{\text{primary}}|} \sum_{m \in \mathcal{M}_{\text{primary}}} \text{Rank}_m(c)$$

The optimal variant $c^*$ is defined by:
$$c^* = \arg\min_{c \in \mathcal{C}} \bar{R}(c)$$

### 1.3 Tie-Breaking Rule
If two or more configurations achieve an identical Mean Rank $\bar{R}(c)$, ties are broken deterministically by **lowest mean inference latency** on the 120 dev pairs:
$$c_{\text{tie-winner}} = \arg\min_{c \in \text{Ties}} \text{Mean Latency}(c)$$

### 1.4 Non-Negotiable Hard Stopping Rule (Null Hypothesis Retention)
The baseline prototype configuration:
$$\text{Baseline: } T=20, \quad \gamma=0.5, \quad K=3, \quad r=5, \quad \text{normalization}=\text{linear}$$
is strictly included in the competition pool $\mathcal{C}$.

**If no candidate configuration achieves a strictly better (lower) mean rank than the baseline prototype ($\bar{R}(c^*) \ge \bar{R}(c_{\text{base}})$), the baseline prototype will be retained as CADM-MSWN v2 without modification, and this outcome will be reported transparently.** No post-hoc metric substitution, re-weighting, or selective filtering is permitted.

---

## 2. Experimental Design & Sweep Clarification: 15 Settings vs. 12 Runs

### 2.1 The 15 Discrete Parameter Settings
The experimental space explores 5 fundamental algorithmic dimensions comprising 15 discrete parameter choices:
1. **Edge compensation coefficient ($\gamma$)**: $\{0.0, 0.25, 0.5\}$ (3 settings)
2. **Weight smoothing temperature ($\tau$)**: $\{0.1, 0.5, 1.0\}$ (3 settings)
3. **Guided filter spatial radius ($r$)**: $\{5, 9, 13\}$ (3 settings)
4. **Diffusion iterations ($T$)**: $\{5, 10, 20\}$ (3 settings)
5. **Cluster regime count ($K$)**: $\{1, 2, 3\}$ (3 settings)

Total parameter values across the 5 dimensions: $3 \times 5 = 15$ discrete settings.

### 2.2 Why 12 Runs (One-at-a-Time Sweep Around the Reconstruction Anchor)
A full factorial grid would entail $3^5 = 243$ configurations $\times 120$ pairs = $29,160$ fusions, which is computationally prohibitive on CPU.  
Instead, the design follows an **informative One-at-a-Time (OAT) sweep centered around the clean reconstruction anchor point** ($\gamma=0.0$, linear weighting, $r=5, T=20, K=3$), alongside the baseline prototype ($\gamma=0.5$):

| Config ID | Configuration Name | $T$ | $K$ | $\gamma$ | Normalization | Radius $r$ | Role in Sweep |
|---|---|---|---|---|---|---|---|
| **C1** | `CADM_v1_prototype` | 20 | 3 | 0.50 | linear | 5 | Baseline prototype reference ($\gamma=0.5$) |
| **C2** | `gamma_0.25` | 20 | 3 | 0.25 | linear | 5 | $\gamma$-sweep intermediate |
| **C3** | `gamma_0.0_no_edge_comp` | 20 | 3 | 0.00 | linear | 5 | **Anchor configuration**: pure $GT+FT$ reconstruction |
| **C4** | `softmax_tau_0.1` | 20 | 3 | 0.00 | softmax ($\tau=0.1$) | 5 | Weight smoothing sweep (sharp softmax) |
| **C5** | `softmax_tau_0.5` | 20 | 3 | 0.00 | softmax ($\tau=0.5$) | 5 | Weight smoothing sweep (moderate softmax) |
| **C6** | `softmax_tau_1.0` | 20 | 3 | 0.00 | softmax ($\tau=1.0$) | 5 | Weight smoothing sweep (smooth softmax) |
| **C7** | `guided_radius_9` | 20 | 3 | 0.00 | linear | 9 | Guided filter radius sweep ($19 \times 19$ window) |
| **C8** | `guided_radius_13` | 20 | 3 | 0.00 | linear | 13 | Guided filter radius sweep ($27 \times 27$ window) |
| **C9** | `T_5` | 5 | 3 | 0.00 | linear | 5 | Iteration sweep: low scale separation |
| **C10** | `T_10` | 10 | 3 | 0.00 | linear | 5 | Iteration sweep: medium scale separation |
| **C11** | `K_1_no_clustering` | 20 | 1 | 0.00 | linear | 5 | Cluster sweep: unclustered global weights |
| **C12** | `K_2` | 20 | 2 | 0.00 | linear | 5 | Cluster sweep: 2-regime segmentation |

*Note on 15 settings $\to$ 12 runs*: The default settings for each factor ($\gamma=0.0, \text{linear}, r=5, T=20, K=3$) serve as the common anchor point (`gamma_0.0_no_edge_comp`), collapsing duplicate evaluations while testing all 15 discrete parameter settings.

---

## 3. Reporting Protocols & Reference Baselines

1. **Reporting Standards**:
   - For all 12 candidate configurations, both **Mean** and **Standard Deviation** across the 120 dev pairs are reported for all 9 metrics:
     - Primary: $Q^{AB/F}, N^{AB/F}, MI, SSIM, VIFP, VIFF$
     - Secondary: $EN, SD, SF, AG$
     - Efficiency: Latency (ms)
   - Zero suppression: Failed or suboptimal configurations are reported with equal fidelity.
2. **Reference Baselines on the Dev Split**:
   - **GFF (Guided Filter Fusion)** (Li et al., 2013) and **Averaging** are evaluated on the exact same 120 dev pairs to anchor the relative performance scale.
