"""
=============================================================================
  Step 2 (BATCH): Train Individual Model (Digital Twin) for ALL Subjects
=============================================================================
Same training logic as the original single-subject script
(2_train_digital_twin.py), but now loops over every subject listed in the
included_subjects.csv produced by the filtering script, instead of training
one hardcoded TARGET_USUBJID.

For each subject it:
  1. Loads the preprocessed CSV from data_processed/<SUBJECT_ID>.csv (or the
     fallback path used by the original script)
  2. Applies the same guard clauses as the original (skip if train/test
     split is empty, or if either split has <1g total carbs)
  3. Trains the IndividualModel (Digital Twin) for N_EPOCHS
  4. Computes the same info.csv metrics (TIR/TAR/TBR/RMSE for AIPop vs AIDT,
     train vs test) and saves them to output_models/<SUBJECT_ID>/

Subjects that fail (missing CSV, guard-clause skip, NaN loss, any
exception) are logged and skipped, so one bad subject does not stop the
whole batch. A summary CSV (batch_training_report.csv) is written at the
end listing status per subject, and a second CSV
(all_subjects_info_summary.csv) concatenates every successfully trained
subject's info.csv into one table for easy analysis across the cohort.

USAGE:
    python train_all_digital_twins.py
    python train_all_digital_twins.py path/to/included_subjects.csv
"""

import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
import torch

from t1dsim_ai.options import hidden_compartments, input_ind
from t1dsim_ai.individual_model import IndividualModel, SequenceSelection
from t1dsim_ai.utils.metrics import (
    get_TIR,
    get_TBR70,
    get_TAR180,
    get_glucose_variability,
)
from t1dsim_ai.utils.preprocess import scale_inverse_Q1

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# CSV produced by the filtering script. Must contain a "subject_id" column
# formatted as T1DEXI-01-XXXX.
INCLUDED_SUBJECTS_CSV = "included_subjects.csv"
if len(sys.argv) > 1:
    INCLUDED_SUBJECTS_CSV = sys.argv[1].strip()

OUTPUT_MODEL_DIR = "output_models/"
BATCH_REPORT_CSV = "batch_training_report.csv"
ALL_INFO_SUMMARY_CSV = "all_subjects_info_summary.csv"

# Directory where preprocessing saved the per-subject CSVs
DATA_PROCESSED_DIR = r"D:\data_processed"

# -----------------------------------------------------------------------
# TEST MODE: Set to True to train only the first subject in the list.
# Set to False (or comment out the line below) to train all subjects.
# -----------------------------------------------------------------------
TEST_MODE = True

# Hyperparameters (matching the paper / the published trainDigitalTwin.py exactly)
N_NEURONS = 128
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
N_EPOCHS = 150         # Total training epochs (e.g. 150)
OVERLAP = 0.9
SEQ_LEN = 60

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers to load the list of subjects to process (same convention as the
# Step 1 batch script)
# ---------------------------------------------------------------------------
def clean_id_from_subject_id(subject_id):
    """'T1DEXI-01-0234' -> '234' (matches the lstrip('0') convention used
    everywhere else in this pipeline)."""
    raw = subject_id.strip().split("-")[-1]
    stripped = raw.lstrip("0")
    return stripped if stripped else raw


def load_subject_list(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run the filtering script first, "
            f"or pass the correct path as a command-line argument."
        )
    df = pd.read_csv(csv_path)
    if "subject_id" not in df.columns:
        raise ValueError(
            f"'{csv_path}' has no 'subject_id' column. Expected the format "
            f"produced by the filtering script (included_subjects.csv)."
        )
    return df["subject_id"].astype(str).tolist()


def resolve_csv_path(subject_id):
    # Primary: where the preprocessing script saves the file
    csv_path = os.path.join(DATA_PROCESSED_DIR, f"{subject_id}.csv")
    if not os.path.exists(csv_path):
        # Fallback: legacy local path used by the original script
        csv_path = os.path.abspath(f"data_processed/{subject_id}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.abspath(
            f"src/t1dsim_ai/models/IndividualModel/{subject_id}/T1DEXIMAIN_{subject_id}.csv"
        )
    return csv_path


# ---------------------------------------------------------------------------
# Core per-subject training (same logic as the original train_subject(),
# now a function that returns a status dict instead of just printing and
# returning early)
# ---------------------------------------------------------------------------
def train_subject(subject_id):
    # Fixed seeds - matches the original trainModel() exactly, so results
    # are reproducible run-to-run for the same subject/data.
    np.random.seed(0)
    torch.manual_seed(0)

    personalization_path = os.path.join(OUTPUT_MODEL_DIR, subject_id)
    os.makedirs(personalization_path, exist_ok=True)

    csv_path = resolve_csv_path(subject_id)

    print("=" * 70)
    print(f" Training Individual Digital Twin Model for Subject {subject_id}")
    print("=" * 70)

    # 1. Load Subject Data
    if not os.path.exists(csv_path):
        return {"status": "SKIPPED_NO_CSV", "reason": f"Dataset file not found at {csv_path}"}

    print(f"\n[1/4] Loading subject dataset: {csv_path}")
    df_subject = pd.read_csv(csv_path)
    print(f"      Loaded {len(df_subject)} records.")
    print(f"      Train points: {sum(df_subject.is_train)}, Test points: {sum(~df_subject.is_train)}")

    # Guard clauses - same as the original trainModel(): skip degenerate
    # subjects rather than training on an empty or carb-less split.
    if len(df_subject[~df_subject.is_train]) == 0 or len(df_subject[df_subject.is_train]) == 0:
        return {"status": "SKIPPED_EMPTY_SPLIT",
                "reason": "Train or test split is empty for this subject."}
    if (df_subject.loc[~df_subject.is_train, "input_meal_carbs"].sum() < 1
            or df_subject.loc[df_subject.is_train, "input_meal_carbs"].sum() < 1):
        return {"status": "SKIPPED_NO_CARBS",
                "reason": "Train or test split has essentially no carbohydrate events (<1g total)."}

    # 2. Setup Neural Network Architecture
    print(f"\n[2/4] Initializing Individual Model Architecture...")
    print(f"      Hidden compartments: {hidden_compartments}")
    print(f"      Epochs: {N_EPOCHS}, Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}")

    nn_individual = IndividualModel(
        subjectID=subject_id,
        df_subj=df_subject,
        pathModel=OUTPUT_MODEL_DIR
    )

    nn_individual.setup_nn(
        hidden_compartments=hidden_compartments,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        overlap=OVERLAP,
        seq_len=SEQ_LEN + 1
    )

    # 3. Fit Model (Training Loop)
    print(f"\n[3/4] Starting PyTorch Training Loop...")
    start_time = time.time()

    loss_result = nn_individual.fit(save_model=True)

    elapsed_minutes = (time.time() - start_time) / 60.0

    if loss_result is np.nan:
        return {"status": "FAILED_NAN_LOSS", "reason": "Training returned NaN loss."}

    # 4. Build the same info.csv content as the original trainDigitalTwin.py
    print(f"\n[4/4] Computing evaluation metrics and saving info.csv...")

    df_info = pd.DataFrame(index=[0])
    df_info["subjectID"] = subject_id
    df_info["TIR"] = 100 * get_TIR(df_subject.output_cgm)
    df_info["TAR"] = 100 * get_TAR180(df_subject.output_cgm)
    df_info["TBR"] = 100 * get_TBR70(df_subject.output_cgm)
    df_info["GlucoseVariability"] = 100 * get_glucose_variability(df_subject.output_cgm)

    data = {
        "train": [
            nn_individual.x_est_train,
            nn_individual.u_pop_train,
            nn_individual.y_id_train,
            nn_individual.u_ind_train,
            nn_individual.cgm_real_train,
        ],
        "test": [
            nn_individual.x_est_test,
            nn_individual.u_pop_test,
            nn_individual.y_id_test,
            nn_individual.u_ind_test,
            nn_individual.cgm_real_test,
        ],
    }

    for group in ["train", "test"]:
        batch = SequenceSelection(SEQ_LEN + 1, "cpu", data[group])
        with torch.no_grad():
            (batch_x0_hidden, batch_u_pop, batch_u_ind, batch_y, batch_x_original) = batch.get_all("all")

            df_info["n_seq_" + group] = batch_x0_hidden.shape[0]

            df_info["TIR_" + group] = 100 * np.mean(np.apply_along_axis(
                get_TIR, 0,
                scale_inverse_Q1(batch_x_original[1:, :, [0]], nn_individual.popModelFolder).detach().numpy()
            ))
            df_info["TAR_" + group] = 100 * np.mean(np.apply_along_axis(
                get_TAR180, 0,
                scale_inverse_Q1(batch_x_original[1:, :, [0]], nn_individual.popModelFolder).detach().numpy()
            ))
            df_info["TBR_" + group] = 100 * np.mean(np.apply_along_axis(
                get_TBR70, 0,
                scale_inverse_Q1(batch_x_original[1:, :, [0]], nn_individual.popModelFolder).detach().numpy()
            ))

            for model_bool in [False, True]:
                model_name = "AIPop" if not model_bool else "AIDT"
                batch_x_sim = nn_individual.nn_solution(batch_x0_hidden, batch_u_pop, batch_u_ind, model_bool)

                df_info["RMSE_" + model_name + "_" + group] = torch.mean(torch.sqrt(torch.nanmean(
                    (scale_inverse_Q1(batch_x_sim[1:, :, [0]], nn_individual.popModelFolder)
                     - scale_inverse_Q1(batch_x_original[1:, :, [0]], nn_individual.popModelFolder)) ** 2,
                    dim=0
                ))).item()
                df_info["TIR_" + model_name + "_" + group] = 100 * np.mean(np.apply_along_axis(
                    get_TIR, 0,
                    scale_inverse_Q1(batch_x_sim[1:, :, [0]], nn_individual.popModelFolder).detach().numpy()
                ))
                df_info["TAR_" + model_name + "_" + group] = 100 * np.mean(np.apply_along_axis(
                    get_TAR180, 0,
                    scale_inverse_Q1(batch_x_sim[1:, :, [0]], nn_individual.popModelFolder).detach().numpy()
                ))
                df_info["TBR_" + model_name + "_" + group] = 100 * np.mean(np.apply_along_axis(
                    get_TBR70, 0,
                    scale_inverse_Q1(batch_x_sim[1:, :, [0]], nn_individual.popModelFolder).detach().numpy()
                ))

    df_info["train_epochs"] = nn_individual.curr_epoch
    info_csv_path = os.path.join(nn_individual.pathModel, "info.csv")
    df_info.T.to_csv(info_csv_path)

    print("=" * 70)
    print(f" SUCCESS! Individual Model Trained in {elapsed_minutes:.2f} minutes.")
    print(f" Final Simulation Loss (RMSE): {loss_result:.6f}")
    print(f" Saved Model Checkpoint to: {OUTPUT_MODEL_DIR}{subject_id}/individual_model.pt")
    print(f" Saved Metrics Summary to: {info_csv_path}")
    print("=" * 70)

    return {
        "status": "OK",
        "reason": "",
        "elapsed_minutes": elapsed_minutes,
        "final_rmse": float(loss_result),
        "info_csv_path": info_csv_path,
        "info_row": df_info,  # kept in-memory for the aggregated summary
    }


# ---------------------------------------------------------------------------
# Batch driver: loop over every subject in included_subjects.csv
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f" Loading subject list from: {INCLUDED_SUBJECTS_CSV}")
    print("=" * 70)
    subject_ids = load_subject_list(INCLUDED_SUBJECTS_CSV)

    if TEST_MODE:
        # User can add any specific subjects to this list to test them:
        subject_ids = ["T1DEXI-01-0043"]
        print(f" [TEST MODE] Training only the specified subjects: {subject_ids}\n")

    results = []
    info_rows = []

    for i, subject_id in enumerate(subject_ids, start=1):
        print(f"\n>>> [{i}/{len(subject_ids)}] Subject {subject_id}")
        try:
            result = train_subject(subject_id)
            status = result["status"]

            if status == "OK":
                info_rows.append(result["info_row"])
                results.append({
                    "subject_id": subject_id,
                    "status": "OK",
                    "reason": "",
                    "elapsed_minutes": round(result["elapsed_minutes"], 2),
                    "final_rmse": result["final_rmse"],
                    "info_csv_path": result["info_csv_path"],
                })
            else:
                print(f"      [SKIPPED/{status}] {subject_id}: {result['reason']}")
                results.append({
                    "subject_id": subject_id,
                    "status": status,
                    "reason": result["reason"],
                    "elapsed_minutes": "",
                    "final_rmse": "",
                    "info_csv_path": "",
                })
        except Exception as e:
            print(f"      [FAILED] {subject_id}: {e}")
            results.append({
                "subject_id": subject_id,
                "status": "FAILED_EXCEPTION",
                "reason": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
                "elapsed_minutes": "",
                "final_rmse": "",
                "info_csv_path": "",
            })
            continue

    report = pd.DataFrame(results)
    report.to_csv(BATCH_REPORT_CSV, index=False)

    if info_rows:
        all_info = pd.concat(info_rows, ignore_index=True)
        all_info.to_csv(ALL_INFO_SUMMARY_CSV, index=False)
    else:
        all_info = pd.DataFrame()

    n_ok = (report["status"] == "OK").sum()
    n_skipped = report["status"].str.startswith("SKIPPED").sum()
    n_failed = report["status"].str.startswith("FAILED").sum()

    print("\n" + "=" * 70)
    print(" BATCH TRAINING SUMMARY")
    print("=" * 70)
    print(f" Total subjects attempted: {len(report)}")
    print(f" Trained successfully:     {n_ok}")
    print(f" Skipped (guard clauses):  {n_skipped}")
    print(f" Failed (exceptions/NaN):  {n_failed}")
    print(f" Per-subject report saved to: {BATCH_REPORT_CSV}")
    if len(all_info):
        print(f" Aggregated info.csv summary ({len(all_info)} subjects) saved to: {ALL_INFO_SUMMARY_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
