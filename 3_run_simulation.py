"""
=============================================================================
  Step 3: Run Digital Twin Simulation & Inference for Specified Subject
=============================================================================
This script strictly loads YOUR newly trained individual Digital Twin model 
from output_models/{SUBJECT_ID}/, simulates 1 day of unseen test data, 
and plots a comparison against actual readings.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from t1dsim_ai.individual_model import DigitalTwin

# ---------------------------------------------------------------------------
# CONFIGURATION: Change subject ID or pass as command line argument
# ---------------------------------------------------------------------------
TARGET_USUBJID =  "0043"         # <-- Change subject ID here! (e.g. 0102, 0692, 0794, 0880, 1047)

# Which 1-day window of the test set to simulate (0 = first day, 1 = second day, ...).
# Set to None to auto-pick the day with the most variety in heart_rate/sleep_efficiency.
# Set to "ALL" to simulate the entire test set (all days combined).
SIM_DAY_INDEX =None

if len(sys.argv) > 1:
    TARGET_USUBJID = sys.argv[1].strip()

if len(sys.argv) > 2:
    arg_day = sys.argv[2].strip()
    SIM_DAY_INDEX = None if arg_day.lower() == "auto" else int(arg_day)

CLEAN_TARGET = TARGET_USUBJID.lstrip('0') if TARGET_USUBJID.lstrip('0') else TARGET_USUBJID
SUBJECT_ID = f"T1DEXI-01-{CLEAN_TARGET.zfill(4)}"

# CSV Dataset Path
DATA_PROCESSED_DIR = r"D:\data_processed"
CSV_PATH = os.path.join(DATA_PROCESSED_DIR, f"{SUBJECT_ID}.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = f"data_processed/{SUBJECT_ID}.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = f"src/t1dsim_ai/models/IndividualModel/{SUBJECT_ID}/T1DEXIMAIN_{SUBJECT_ID}.csv"

# YOUR TRAINED MODEL PATH (Strictly from output_models/)
YOUR_TRAINED_MODEL_DIR = os.path.abspath(f"output_models/{SUBJECT_ID}/")

OUTPUT_FIG = f"simulation_{SUBJECT_ID}.png"

def run_simulation():
    print("=" * 70)
    print(f" Step 3: Running Digital Twin Simulation for Subject {SUBJECT_ID}")
    print("=" * 70)

    # 1. Verify YOUR Trained Model Exists
    pt_file = os.path.join(YOUR_TRAINED_MODEL_DIR, "individual_model.pt")
    pkl_file = os.path.join(YOUR_TRAINED_MODEL_DIR, "scaler_robust.pkl")

    if not os.path.exists(pt_file) or not os.path.exists(pkl_file):
        print(f"ERROR: YOUR trained model files were not found in:")
        print(f"       {YOUR_TRAINED_MODEL_DIR}")
        print(f"\nPlease train your model first by running Step 2:")
        print(f"       python 2_train_individual_model.py {CLEAN_TARGET}")
        return

    print(f"\n[1/3] Loading YOUR trained model files:")
    print(f"      Model weights: {pt_file}")
    print(f"      Scaler file:  {pkl_file}")

    # 2. Load Data & Filter Test Split
    print(f"\n[2/3] Loading dataset: {CSV_PATH}")
    df_data = pd.read_csv(CSV_PATH)
    
    # Ensure column compatibility across formats
    if 'output_cgm' not in df_data.columns and 'cgm' in df_data.columns:
        df_data['output_cgm'] = df_data['cgm']
    
    if 'input_meal_carbs' not in df_data.columns:
        if 'input_meal_carbs_ODEPop' in df_data.columns:
            df_data['input_meal_carbs'] = df_data['input_meal_carbs_ODEPop']
        elif 'carbs' in df_data.columns:
            df_data['input_meal_carbs'] = df_data['carbs']
        else:
            df_data['input_meal_carbs'] = 0.0

    if 'input_insulin' not in df_data.columns:
        if 'input_insulin_ODEPop' in df_data.columns:
            df_data['input_insulin'] = df_data['input_insulin_ODEPop']
        elif 'insulin_rate_basal' in df_data.columns:
            df_data['input_insulin'] = df_data['insulin_rate_basal']
        else:
            df_data['input_insulin'] = 0.0

    df_test = df_data[~df_data.is_train].copy().reset_index(drop=True)
    n_test = len(df_test)
    n_days_available = n_test // 288

    if n_test < 288:
        print("Warning: Test set has less than 288 samples (1 day). Using available test data.")
        sim_window = df_test
        chosen_day = 0
    else:
        if SIM_DAY_INDEX == "ALL":
            sim_window = df_test
            chosen_day = "ALL"
            print(f"      Selected test simulation window: ALL {len(sim_window)} samples (approx {n_days_available} days).")
        elif SIM_DAY_INDEX is None:
            # Auto-pick the day with the most variety in heart_rate/sleep_efficiency,
            # so the plot doesn't land on a flat/imputed day by chance.
            best_score, chosen_day = -1, 0
            for day in range(n_days_available):
                w = df_test.iloc[day * 288 : (day + 1) * 288]
                score = 0
                if "heart_rate" in w.columns:
                    score += w["heart_rate"].nunique()
                if "sleep_efficiency" in w.columns:
                    score += w["sleep_efficiency"].nunique()
                if score > best_score:
                    best_score, chosen_day = score, day
            print(f"      Auto-selected day {chosen_day} (most variety in HR/sleep, score={best_score}).")
            sim_window = df_test.iloc[chosen_day * 288 : (chosen_day + 1) * 288]
        else:
            chosen_day = min(SIM_DAY_INDEX, max(n_days_available - 1, 0))
            if chosen_day != SIM_DAY_INDEX:
                print(f"      Warning: SIM_DAY_INDEX={SIM_DAY_INDEX} out of range, using day {chosen_day} instead.")
            sim_window = df_test.iloc[chosen_day * 288 : (chosen_day + 1) * 288]

    if chosen_day != "ALL":
        print(f"      Selected test simulation window: {len(sim_window)} samples (day index {chosen_day} of {n_days_available} available).")

    # 3. Instantiate DigitalTwin with YOUR trained model directory
    print(f"\n[3/3] Simulating forward trajectory with YOUR model...")
    my_dt = DigitalTwin(custom_DT=YOUR_TRAINED_MODEL_DIR)
    df_sim = my_dt.simulate(sim_window)

    # 4. Visualization (matches runDigitalTwin.py style exactly)
    color_AIDT = "#0072B2"
    color_AIPop = "#009E73"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    time = np.arange(len(df_sim))

    ax1.plot(time, df_sim.cgm_Actual, ms=5, c="k")
    ax1.plot(time, df_sim.cgm_NNPop, ".", ms=5, c=color_AIPop)
    ax1.plot(time, df_sim.cgm_NNDT, ".", ms=5, c=color_AIDT)

    ax1.plot(-1, -1, "o", ms=5, c="k", label="Actual")
    ax1.plot(-1, -1, "o", ms=5, c=color_AIPop, label="NN-based population")
    ax1.plot(-1, -1, "o", ms=5, c=color_AIDT, label="NN-based digital twin")

    ax1.axhspan(70, 180, facecolor="gray", alpha=0.1)

    for y in [70, 180, 250]:
        ax1.axhline(y, color="k", alpha=0.2, lw=0.3)

    ax1.legend(loc=9, ncol=3, frameon=False)
    ax1.set_ylim(40, 380)
    ax1.set_ylabel("CGM [mg/dL]")
    for location in ["left", "right", "top", "bottom"]:
        ax1.spines[location].set_linewidth(0.1)

    ax2.plot(time, df_sim.input_insulin, c="k", lw=0.7)
    ax2.set_ylim(-1, max(65, df_sim.input_insulin.max() * 1.2))

    ax2.set_ylabel("Insulin [U/h]")
    for location in ["left", "right", "top", "bottom"]:
        ax2.spines[location].set_linewidth(0.1)
    ax2.spines["top"].set_linewidth(0.5)

    ax2_carbs = ax2.twinx()
    color = "tab:red"
    ax2_carbs.plot(
        df_sim.loc[df_sim.input_meal_carbs != 0].index,
        df_sim.loc[df_sim.input_meal_carbs != 0, "input_meal_carbs"],
        "o",
        color=color,
    )
    ax2_carbs.set_ylim(-1, max(65, df_sim.input_meal_carbs.max() * 1.2))

    ax2_carbs.set_ylabel("Meal carbs [g]", color=color)
    ax2_carbs.tick_params(axis="y", labelcolor=color)
    ax2_carbs.spines["right"].set_position(("axes", 0))
    for location in ["left", "right", "top", "bottom"]:
        ax2_carbs.spines[location].set_linewidth(0)

    # Optional extra channels (only plotted if present in the data, same as runDigitalTwin.py)
    if "heart_rate" in df_sim.columns:
        ax2_hr = ax2.twinx()
        color = "tab:green"
        ax2_hr.plot(time, df_sim.heart_rate, lw=0.5, color=color)
        ax2_hr.set_ylim(-1.4, 147)
        ax2_hr.set_ylabel("Heart rate [BPM]", color=color)
        ax2_hr.tick_params(axis="y", labelcolor=color)
        for location in ["left", "right", "top", "bottom"]:
            ax2_hr.spines[location].set_linewidth(0)

    if "sleep_efficiency" in df_sim.columns:
        ax2_pa = ax2.twinx()
        color = "tab:purple"
        ax2_pa.plot(time, df_sim.sleep_efficiency, lw=1, color=color)
        ax2_pa.set_ylim(-0.01, 1.05)
        for location in ["left", "right", "top", "bottom"]:
            ax2_pa.spines[location].set_linewidth(0)
        ax2_pa.set_ylabel("Sleep efficiency", color=color, labelpad=-20)
        ax2_pa.tick_params(axis="y", labelcolor=color)
        ax2_pa.set_yticks([0, 1], [0, 1])
        ax2_pa.tick_params(axis="y", direction="in", pad=-15)

    n_steps = len(df_sim)
    day_frac = n_steps / (12 * 24) if n_steps else 1
    ax2.set_xticks(
        np.arange(0, n_steps + 1, 12 * 12),
        np.round(np.arange(0, day_frac + 0.01, 0.5), 2),
    )
    ax2.set_xlabel("Simulation time [day]")
    ax2.set_xlim(-12 * 3, n_steps + 12 * 2)

    plt.subplots_adjust(hspace=0)

    plt.savefig(OUTPUT_FIG, dpi=500, bbox_inches="tight")
    print("=" * 70)
    print(f" SUCCESS! Simulation Plot saved using YOUR model to: {OUTPUT_FIG}")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()
