"""
=============================================================================
  Find and Simulate Top Patients with High-Quality Wearable Data (Like 0692)
=============================================================================
This script identifies and runs simulations for the top 3 patients whose 
data is rich, dynamic, and complete (continuous Polar heart rate, active sleep
transitions, and logged meals) exactly like patient T1DEXI-01-0692.
"""

import os
import subprocess
import pandas as pd
import numpy as np

# The 3 patients with complete 24/7 continuous wearable sensors & meal logging
TOP_PATIENTS = ["0102", "0794", "1047"]

def inspect_patient_data():
    print("=" * 75)
    print("  PATIENT DATA QUALITY COMPARISON (Continuity & Dynamism)")
    print("=" * 75)
    
    summary = []
    for pid in ["0692"] + TOP_PATIENTS:
        subj = f"T1DEXI-01-{pid}"
        csv_path = f"src/t1dsim_ai/models/IndividualModel/{subj}/T1DEXIMAIN_{subj}.csv"
        
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        test = df[~df.is_train].reset_index(drop=True)
        
        # Check first test day metrics
        day_sample = test.iloc[288:576]
        hr = day_sample["heart_rate"]
        
        carbs_col = "input_meal_carbs_ODEPop" if "input_meal_carbs_ODEPop" in day_sample else "input_meal_carbs"
        n_meals = (day_sample[carbs_col] > 0).sum()
        total_carbs = day_sample[carbs_col].sum()
        
        summary.append({
            "Patient": subj,
            "HR Unique Readings/Day": hr.nunique(),
            "HR Min (BPM)": round(hr.min(), 1),
            "HR Max (BPM)": round(hr.max(), 1),
            "HR Std": round(hr.std(), 1),
            "Meals/Day": n_meals,
            "Total Carbs (g)": round(total_carbs, 1),
            "Sleep States": day_sample["sleep_efficiency"].nunique()
        })
        
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    print("=" * 75)
    return summary

def run_all_simulations():
    print("\nRunning simulations for the 3 selected high-quality patients...\n")
    for pid in TOP_PATIENTS:
        subj = f"T1DEXI-01-{pid}"
        print(f"--> Simulating {subj} ...")
        cmd = ["python", "3_run_simulation.py", pid]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"    [SUCCESS] Saved: simulation_{subj}.png")
        else:
            print(f"    [ERROR] Failed to simulate {subj}:\n{res.stderr}")

if __name__ == "__main__":
    inspect_patient_data()
    run_all_simulations()
