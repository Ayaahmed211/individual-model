"""
=============================================================================
  Diagnose Raw Heart Rate (VS.xpt) & Find Patients with HR Like 0692
=============================================================================
Analyzes raw heart-rate signals BEFORE any preprocessing is performed.
Uses Patient T1DEXI-01-0692 as the gold-standard benchmark (continuous 24/7
wearable watch data with ~208k HRM samples) and finds other subjects with 
identical raw wearable coverage.

USAGE:
  1. Find subjects in raw data matching 0692 (instant):
       python diagnose_hr.py
       python diagnose_hr.py --find-similar

  2. Inspect raw data for a specific subject (e.g. 0692, 0781, 1493):
       python diagnose_hr.py 0781
       python diagnose_hr.py 1493 --deep
"""

import os
import sys
import time
import pandas as pd
import numpy as np

RAW_DATA_DIR = r"D:\T1DEXI - DATA FOR UPLOAD\T1DEXI - DATA FOR UPLOAD"
FILE_PATH_OVERRIDES = {
    "VS.xpt": r"C:\Users\3com\OneDrive\Desktop\VS.xpt",
}
SUMMARY_CACHE_CSV = "raw_hr_coverage_summary.csv"
INCLUDED_CSV = "included_subjects.csv"

# Patient 0692 Gold-Standard Benchmark (from raw VS.xpt)
BENCHMARK_0692 = {
    "subject_id": "T1DEXI-01-0692",
    "clean_id": "692",
    "raw_hrm_records": 208299,
    "distinct_bpm_values": 125,
    "sampling_rate_sec": 10.0,
    "time_span_days": 51.0,
    "median_gap_min": 0.2,
}

def decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)

def get_vs_path():
    return FILE_PATH_OVERRIDES.get("VS.xpt", os.path.join(RAW_DATA_DIR, "VS.xpt"))

def print_benchmark_card():
    print("=" * 78)
    print("  PATIENT 0692 RAW HEART RATE BENCHMARK (Before Preprocessing)")
    print("=" * 78)
    print(f"  Subject:                {BENCHMARK_0692['subject_id']}")
    print(f"  Continuous HRM Samples: {BENCHMARK_0692['raw_hrm_records']:,} records (~10-second intervals)")
    print(f"  Distinct BPM Levels:    {BENCHMARK_0692['distinct_bpm_values']} distinct values (range 34 - 180 BPM)")
    print(f"  Sampling Interval:      {BENCHMARK_0692['median_gap_min']} min (10 seconds - Verily Watch)")
    print(f"  Total Wear Coverage:    {BENCHMARK_0692['time_span_days']:.1f} days of continuous wearable data")
    print("  Why 0692 is dynamic in simulation:")
    print("    -> 24/7 continuous wearable sensor with zero large gaps.")
    print("    -> Realistic fluctuations between sleep (40 BPM) and activity (140 BPM).")
    print("=" * 78)

def ensure_summary_cache(vs_path):
    if os.path.exists(SUMMARY_CACHE_CSV):
        return pd.read_csv(SUMMARY_CACHE_CSV)
    
    print(f"\nBuilding raw HR coverage cache from {vs_path}...")
    print("This scans all subjects in VS.xpt and will be saved for instant future runs.\n")
    t0 = time.time()
    reader = pd.read_sas(vs_path, format="xport", chunksize=1000000)
    
    hrm_counts = {}
    hr_counts = {}
    
    for i, chunk in enumerate(reader, 1):
        sub = chunk[["USUBJID", "VSTESTCD"]].dropna()
        sub["USUBJID"] = sub["USUBJID"].apply(decode).str.strip()
        sub["VSTESTCD"] = sub["VSTESTCD"].apply(decode).str.strip()
        
        hrm = sub[sub["VSTESTCD"] == "HRM"]
        if len(hrm):
            for sid, c in hrm["USUBJID"].value_counts().items():
                clean = sid.lstrip("0") or sid
                hrm_counts[clean] = hrm_counts.get(clean, 0) + c
                
        hr = sub[sub["VSTESTCD"] == "HR"]
        if len(hr):
            for sid, c in hr["USUBJID"].value_counts().items():
                clean = sid.lstrip("0") or sid
                hr_counts[clean] = hr_counts.get(clean, 0) + c
                
        if i % 10 == 0:
            print(f"  {i*1000000:,} rows scanned | {len(hrm_counts)} subjects found | {round(time.time()-t0, 1)}s")

    all_sids = set(hrm_counts.keys()) | set(hr_counts.keys())
    rows = []
    for sid in all_sids:
        c_hrm = hrm_counts.get(sid, 0)
        c_hr = hr_counts.get(sid, 0)
        rows.append({
            "clean_id": str(sid),
            "subject_id": f"T1DEXI-01-{str(sid).zfill(4)}",
            "raw_hrm_records": c_hrm,
            "raw_hr_records": c_hr,
            "total_hr_records": c_hrm + c_hr
        })

    df_out = pd.DataFrame(rows).sort_values(by="raw_hrm_records", ascending=False)
    df_out.to_csv(SUMMARY_CACHE_CSV, index=False)
    print(f"\n[DONE] Saved coverage summary for {len(df_out)} subjects to {SUMMARY_CACHE_CSV} in {round(time.time()-t0, 1)}s.\n")
    return df_out

def find_similar_to_0692():
    print_benchmark_card()
    vs_path = get_vs_path()
    df_sum = ensure_summary_cache(vs_path)
    df_sum["clean_id"] = df_sum["clean_id"].astype(str)
    
    # Load included subjects to know which ones qualify for simulation
    inc_ids = []
    if os.path.exists(INCLUDED_CSV):
        df_inc = pd.read_csv(INCLUDED_CSV)
        inc_ids = df_inc["subject_id"].tolist()
        
    df_sum["in_study"] = df_sum["subject_id"].isin(inc_ids)
    
    b_hrm = BENCHMARK_0692["raw_hrm_records"]
    df_sum["diff_from_0692"] = (df_sum["raw_hrm_records"] - b_hrm).abs()
    df_sum["pct_similarity"] = 100.0 - (df_sum["diff_from_0692"] / b_hrm * 100.0)
    df_sum["pct_similarity"] = df_sum["pct_similarity"].clip(lower=0.0)
    
    print("\n" + "=" * 78)
    print("  TOP RAW PATIENTS CLOSEST TO 0692 IN WEARABLE COVERAGE (HRM)")
    print("=" * 78)
    
    # Filter to included study patients
    matches = df_sum[df_sum["in_study"]].sort_values(by="diff_from_0692").head(15)
    
    display_rows = []
    for rank, (_, row) in enumerate(matches.iterrows(), 1):
        is_0692 = " (BENCHMARK)" if str(row["clean_id"]) == "692" else ""
        display_rows.append({
            "Rank": rank,
            "Subject ID": f"{row['subject_id']}{is_0692}",
            "Raw HRM Samples": f"{int(row['raw_hrm_records']):,}",
            "Polar HR Samples": f"{int(row['raw_hr_records']):,}",
            "Similarity to 0692": f"{row['pct_similarity']:.2f}%",
            "Study Qualified": "YES" if row["in_study"] else "NO"
        })
        
    df_disp = pd.DataFrame(display_rows)
    print(df_disp.to_string(index=False))
    print("=" * 78)
    
    print("\n[RECOMMENDATION FOR PREPROCESSING & SIMULATION]")
    top_3 = matches[matches["clean_id"] != "692"].head(3)
    p_ids = top_3["clean_id"].tolist()
    print(f"The top 3 patients in the raw dataset with heart rate coverage like 0692 are:")
    for idx, (_, r) in enumerate(top_3.iterrows(), 1):
        print(f"  {idx}. Subject {r['subject_id']} -> {int(r['raw_hrm_records']):,} raw HRM samples ({r['pct_similarity']:.1f}% match)")
    print(f"\nYou can run diagnose on any of them individually:")
    print(f"  python diagnose_hr.py {p_ids[0]}")
    print(f"  python diagnose_hr.py {p_ids[1]}")
    print(f"  python diagnose_hr.py {p_ids[2]}")

def diagnose_single_subject(target, deep_scan=False):
    target_clean = target.lstrip("0") or target
    target_full = f"T1DEXI-01-{target_clean.zfill(4)}"
    
    print_benchmark_card()
    print(f"\n  DIAGNOSING SUBJECT: {target_full} (USUBJID: {target_clean})")
    print("=" * 78)
    
    # 1. Instant Cache Lookup
    vs_path = get_vs_path()
    df_sum = ensure_summary_cache(vs_path)
    df_sum["clean_id"] = df_sum["clean_id"].astype(str)
    row = df_sum[df_sum["clean_id"] == target_clean]
    
    if len(row) == 0:
        print(f"Warning: Subject {target_clean} not found in raw VS.xpt summary.")
        return
        
    c_hrm = int(row["raw_hrm_records"].values[0])
    c_hr = int(row["raw_hr_records"].values[0])
    total = int(row["total_hr_records"].values[0])
    
    diff = abs(c_hrm - BENCHMARK_0692["raw_hrm_records"])
    sim = max(0.0, 100.0 - (diff / BENCHMARK_0692["raw_hrm_records"] * 100.0))
    
    print(f"  Continuous Wearable (HRM) Samples: {c_hrm:,}")
    print(f"  Exercise Polar HR Samples:         {c_hr:,}")
    print(f"  Total Heart Rate Samples in VS:    {total:,}")
    print(f"  Similarity to 0692 Coverage:       {sim:.2f}%")
    
    if os.path.exists(INCLUDED_CSV):
        inc_ids = pd.read_csv(INCLUDED_CSV)["subject_id"].tolist()
        in_study = target_full in inc_ids
        print(f"  Qualified in included_subjects:    {'YES' if in_study else 'NO'}")
        
    print("-" * 78)
    if c_hrm >= 180000:
        print("  VERDICT: EXCELLENT! Matches 0692's gold standard wearable coverage.")
        print("           Simulation will produce active, wavy, dynamic curves with real BPM changes.")
    elif c_hrm >= 80000:
        print("  VERDICT: MODERATE. Has wearable data but with some missing days/gaps.")
    else:
        print("  VERDICT: SPARSE. Very low watch data (will produce flat horizontal lines).")
    print("=" * 78)
    
    if deep_scan:
        print("\nRunning deep scan of VS.xpt for detailed timestamps and gaps...")
        CHUNK = 200_000
        matched = []
        reader = pd.read_sas(vs_path, format="xport", chunksize=CHUNK)
        for chunk in reader:
            sid = chunk["USUBJID"].apply(decode).astype(str).str.lstrip("0")
            hit = chunk[sid == target_clean]
            if len(hit):
                matched.append(hit)
            elif matched:
                break
        if matched:
            df = pd.concat(matched, ignore_index=True)
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].apply(decode)
            df["start"] = pd.to_datetime(df["VSDTC"], unit="s", errors="coerce")
            for code, g in df.groupby("VSTESTCD"):
                print(f"  VSTESTCD = {code!r}: {len(g):,} records | Range: {g['VSSTRESN'].min():.0f}-{g['VSSTRESN'].max():.0f} BPM | Distinct: {g['VSSTRESN'].nunique()}")

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].strip() in ["--find-similar", "--similar", "--all", "-a"]:
        find_similar_to_0692()
    else:
        target = sys.argv[1].strip()
        deep = "--deep" in sys.argv
        diagnose_single_subject(target, deep_scan=deep)
