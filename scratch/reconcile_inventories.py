import os
import sys
import json
import hashlib
from pathlib import Path

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def reconcile():
    print("=== Reconciling dataset inventories ===")
    
    # 1. TNO detailed audit
    tno_dir = Path("data/raw/tno")
    tno_subdirs = sorted([d for d in tno_dir.iterdir() if d.is_dir() and d.name not in ["thermal", "visual"]])
    print(f"Found {len(tno_subdirs)} TNO scene subdirectories")
    
    tno_pairs = []
    for d in tno_subdirs:
        files = list(d.glob("*.bmp")) + list(d.glob("*.png")) + list(d.glob("*.jpg"))
        ir_file = None
        vi_file = None
        
        # Priority rules for IR vs VI in TNO
        for f in files:
            name_lower = f.name.lower()
            if any(k in name_lower for k in ["ir", "lwir", "nir", "thermal"]):
                if ir_file is None or "lwir" in name_lower or "37rad" in name_lower:
                    ir_file = f
            elif any(k in name_lower for k in ["photo", "daylight", "visual", "vis", "tv"]):
                vi_file = f
                
        # If not explicitly named, fallback to first/second
        if ir_file is None or vi_file is None:
            if len(files) == 2:
                # Typically file 0 is IR, file 1 is VIS or vice versa
                ir_file = files[0]
                vi_file = files[1]
            elif len(files) > 2:
                for f in files:
                    if "rad" in f.name.lower() or "ir" in f.name.lower():
                        ir_file = f
                    elif "dhv" in f.name.lower() or "photo" in f.name.lower() or "vis" in f.name.lower():
                        vi_file = f
                        
        if ir_file and vi_file and ir_file != vi_file:
            tno_pairs.append({
                "pair_id": f"tno_{d.name}",
                "source1": str(ir_file),
                "source2": str(vi_file),
                "source1_sha256": get_sha256(ir_file),
                "source2_sha256": get_sha256(vi_file),
            })
        else:
            print(f"Warning: Could not form pair for {d.name}, files: {[f.name for f in files]}")
            
    print(f"Total verified TNO pairs formed: {len(tno_pairs)}")
    
    # 2. MSRS test detailed audit
    msrs_test_ir = Path("data/raw/msrs/test/ir")
    msrs_test_vi = Path("data/raw/msrs/test/vi")
    msrs_pairs = []
    
    if msrs_test_ir.exists() and msrs_test_vi.exists():
        ir_files = sorted(list(msrs_test_ir.glob("*.png")))
        for ir_p in ir_files:
            vi_p = msrs_test_vi / ir_p.name
            if vi_p.exists():
                msrs_pairs.append({
                    "pair_id": f"msrs_test_{ir_p.stem}",
                    "source1": str(ir_p),
                    "source2": str(vi_p),
                    "source1_sha256": get_sha256(ir_p),
                    "source2_sha256": get_sha256(vi_p),
                })
    print(f"Total verified MSRS test pairs: {len(msrs_pairs)}")
    
    # Load existing dataset_inventory_audit.json
    audit_path = Path("results/dataset_inventory_audit.json")
    with open(audit_path, "r") as f:
        audit_data = json.load(f)
        
    audit_data["tno"] = {
        "count": len(tno_pairs),
        "pairs": tno_pairs[:5] # representative sample entries
    }
    audit_data["tno_full_count"] = len(tno_pairs)
    
    audit_data["msrs"] = {
        "count": len(msrs_pairs),
        "pairs": msrs_pairs[:5] # representative sample entries
    }
    audit_data["msrs_full_count"] = len(msrs_pairs)
    
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Updated and reconciled {audit_path}")
    
    # Check all_datasets_verified_inventory.json
    inv_path = Path("results/all_datasets_verified_inventory.json")
    with open(inv_path, "r") as f:
        inv_data = json.load(f)
    print("\nVerification of all_datasets_verified_inventory.json:")
    for k, v in inv_data["datasets"].items():
        print(f"  {k}: count={v['pair_count']}, expected={v['expected_count']}, status={v['status']}")

if __name__ == "__main__":
    reconcile()
