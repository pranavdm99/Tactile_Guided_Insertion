#!/usr/bin/env python3
import h5py
import numpy as np
import os
import glob

def get_dataset_structure(filepath):
    """Extracts structural metadata and shapes from a single hdf5 file."""
    try:
        with h5py.File(filepath, "r") as f:
            if "data" not in f:
                return None
            demos = list(f["data"].keys())
            if not demos:
                return None
            
            # Use the first demo as the structural template
            first_demo = f[f"data/{demos[0]}"]
            structure = {
                "action_shape": first_demo['actions'].shape[1:],
                "obs_keys": sorted(list(first_demo['obs'].keys())),
                "obs_shapes": {k: first_demo['obs'][k].shape[1:] for k in first_demo['obs'].keys()},
                "num_demos": len(demos)
            }
            return structure
    except Exception as e:
        print(f"  [ERROR] Could not read {os.path.basename(filepath)}: {e}")
        return None

def run_batch_test(dataset_dir):
    files = sorted(glob.glob(os.path.join(dataset_dir, "*.hdf5")) + glob.glob(os.path.join(dataset_dir, "*.h5")))
    
    if not files:
        print(f"❌ No dataset files found in {dataset_dir}")
        return

    # Use 'reference.hdf5' as the gold standard if it exists, otherwise use the first file
    ref_file = next((f for f in files if "reference.hdf5" in f), files[0])
    ref_stats = get_dataset_structure(ref_file)
    
    print(f"🔎 Found {len(files)} files. Comparing against: {os.path.basename(ref_file)}\n")
    
    print("-" * 100)
    print(f"{'FILENAME':<35} | {'DEMOS':<6} | {'STATUS'}")
    print("-" * 100)

    for file_path in files:
        fname = os.path.basename(file_path)
        if file_path == ref_file:
            print(f"{fname:<35} | {ref_stats['num_demos']:<6} | ✅ REFERENCE")
            continue

        stats = get_dataset_structure(file_path)
        if stats is None:
            print(f"{fname:<35} | {'N/A':<6} | ❌ READ ERROR")
            continue

        errors = []
        key_diff_msg = ""
        
        # 1. Check Action Dimensions
        if stats['action_shape'] != ref_stats['action_shape']:
            errors.append(f"ActionDim {stats['action_shape']}")
        
        # 2. Check Observation Keys (The Detailed Fix)
        if stats['obs_keys'] != ref_stats['obs_keys']:
            errors.append("Keys Mismatch")
            missing = set(ref_stats['obs_keys']) - set(stats['obs_keys'])
            extra = set(stats['obs_keys']) - set(ref_stats['obs_keys'])
            if missing: key_diff_msg += f"\n      🔻 MISSING: {list(missing)}"
            if extra:   key_diff_msg += f"\n      🔺 EXTRA  : {list(extra)}"
            
        # 3. Check Image Resolutions
        for k in ref_stats['obs_shapes']:
            if k in stats['obs_shapes'] and stats['obs_shapes'][k] != ref_stats['obs_shapes'][k]:
                errors.append(f"Res Mismatch ({k})")

        if not errors:
            print(f"{fname:<35} | {stats['num_demos']:<6} | ✅ PASS")
        else:
            print(f"{fname:<35} | {stats['num_demos']:<6} | ❌ FAIL ({', '.join(errors)})")
            if key_diff_msg:
                print(key_diff_msg)

    print("-" * 100)

if __name__ == "__main__":
    DATASET_PATH = "/app/datasets"
    run_batch_test(DATASET_PATH)