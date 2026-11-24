import pandas as pd
import numpy as np

def generate_by_dataset_summary():
    per_pair_path = "results/dev_ablation_per_pair.csv"
    df = pd.read_csv(per_pair_path)
    
    # Drop VIFF if present
    if "VIFF" in df.columns:
        df = df.drop(columns=["VIFF"])
        
    metrics = ["QABF", "NABF", "MI", "SSIM", "VIFP", "EN", "SD", "SF", "AG", "Latency_ms"]
    
    # Group by Config and Category
    grouped = df.groupby(["Config", "Category"])[metrics].agg(["mean", "std"]).reset_index()
    
    # Flatten column multiindex
    grouped.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0]
        for c in grouped.columns
    ]
    
    # Round numerical columns
    for c in grouped.columns:
        if c not in ["Config", "Category"]:
            grouped[c] = grouped[c].round(4)
            
    out_path = "results/dev_ablation_by_dataset.csv"
    grouped.to_csv(out_path, index=False)
    print(f"Saved per-dataset summary table to {out_path}")
    print(f"Total rows: {len(grouped)}")

if __name__ == "__main__":
    generate_by_dataset_summary()
