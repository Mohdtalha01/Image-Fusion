"""Summarize Primary and Secondary metrics across all methods on the 5 benchmark pairs.

Identifies wins, ties, and losses for every method across:
- Primary Metrics: Q^AB/F (high), N^AB/F (low), MI (high), SSIM (high), VIFP (high)
- Secondary Metrics: EN, SD, SF, AG
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def main():
    df = pd.read_csv("results/sanity_check/complete_sanity_metrics.csv")
    
    # Pre-registered metric categories: 5 Primary (25 total evaluations across 5 pairs)
    primary_metrics = ["QABF", "NABF", "MI", "SSIM", "VIF"]  # Note: VIF is VIFP
    secondary_metrics = ["EN", "SD", "SF", "AG"]
    
    pairs = df["Pair"].unique()
    methods = df["Method"].unique()
    
    print(f"Total pairs evaluated: {len(pairs)}")
    print(f"Methods: {list(methods)}")
    
    # Calculate wins per metric
    primary_wins = {m: 0 for m in methods}
    secondary_wins = {m: 0 for m in methods}
    
    metric_analysis = []
    
    for p in pairs:
        sub = df[df["Pair"] == p].copy()
        print(f"\n--- Benchmark Pair: {p} ({sub['Modality'].iloc[0]}) ---")
        
        # Primary winners
        # QABF: max
        qabf_winner = sub.loc[sub["QABF"].idxmax()]["Method"]
        # NABF: min (lower is better)
        nabf_winner = sub.loc[sub["NABF"].idxmin()]["Method"]
        # MI: max
        mi_winner = sub.loc[sub["MI"].idxmax()]["Method"]
        # SSIM: max
        ssim_winner = sub.loc[sub["SSIM"].idxmax()]["Method"]
        # VIFP: max
        vif_winner = sub.loc[sub["VIF"].idxmax()]["Method"]
        
        primary_wins[qabf_winner] += 1
        primary_wins[nabf_winner] += 1
        primary_wins[mi_winner] += 1
        primary_wins[ssim_winner] += 1
        primary_wins[vif_winner] += 1
        
        # Secondary winners
        en_winner = sub.loc[sub["EN"].idxmax()]["Method"]
        sd_winner = sub.loc[sub["SD"].idxmax()]["Method"]
        sf_winner = sub.loc[sub["SF"].idxmax()]["Method"]
        ag_winner = sub.loc[sub["AG"].idxmax()]["Method"]
        
        secondary_wins[en_winner] += 1
        secondary_wins[sd_winner] += 1
        secondary_wins[sf_winner] += 1
        secondary_wins[ag_winner] += 1
        
        print(f"  Primary Wins   -> QABF: {qabf_winner} ({sub['QABF'].max():.4f}) | NABF (lowest noise): {nabf_winner} ({sub['NABF'].min():.4f}) | MI: {mi_winner} ({sub['MI'].max():.4f}) | SSIM: {ssim_winner} ({sub['SSIM'].max():.4f}) | VIFP: {vif_winner} ({sub['VIF'].max():.4f})")
        print(f"  Secondary Wins -> EN: {en_winner} ({sub['EN'].max():.4f}) | SD: {sd_winner} ({sub['SD'].max():.4f}) | SF: {sf_winner} ({sub['SF'].max():.4f}) | AG: {ag_winner} ({sub['AG'].max():.4f})")
        
    print("\n=== TOTAL WIN TALLIES ON 5 BENCHMARK PAIRS ===")
    print("Primary Metrics Total Wins (out of 25 possible: 5 metrics x 5 pairs):")
    for m, w in sorted(primary_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m:20s}: {w:2d} / 25 wins ({w/25*100:5.1f}%)")
        
    print("\nSecondary Metrics Total Wins (out of 20 possible: 4 metrics x 5 pairs):")
    for m, w in sorted(secondary_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m:20s}: {w:2d} / 20 wins ({w/20*100:5.1f}%)")

if __name__ == "__main__":
    main()
