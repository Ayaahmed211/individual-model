"""
=============================================================================
  Step 5: Cohort-wide Evaluation (replicates paper's Table 3 methodology)
=============================================================================
Runs the Step-4-style per-window evaluation for EVERY subject that has both
a preprocessed CSV and a trained model, collects one row per subject (mean
TIR/TAR/TBR/LBGI/HBGI/MG for actual vs simulated, averaged across that
subject's independently-simulated 5-hour test windows), then runs the
paper's paired TOST equivalence test treating each subject's mean as one
paired observation across the cohort - matching Section 2.6's "Across 394
digital twins" unit of analysis (Table 3), not a single subject's windows.

FIX in this version: the prediction-RMSE calculation previously compared
the ACTUAL value h steps in the future against the SIMULATED value at the
CURRENT time step (a misaligned shift: errors.append(a_win[steps:] vs
s_win[:-steps])). That doesn't match the paper's Section 2.7.3 methodology,
which anchors both the actual and simulated trajectories at the SAME t=0
initial condition and compares them at the SAME future time point h. Fixed
to compare a_win[steps] against s_win[steps] directly (one error sample per
window per horizon), which is why RMSE now grows with horizon as expected
instead of staying flat around 60 mg/dL regardless of horizon.

Requires: pip install statsmodels

Usage:
    python 5_evaluate_cohort.py
    python 5_evaluate_cohort.py path/to/subject_list.csv
"""

import os
import sys
import glob
import traceback
import numpy as np
import pandas as pd

try:
    from statsmodels.stats.weightstats import ttost_paired
except ImportError:
    print("This script needs statsmodels. Install it with:")
    print("    pip install statsmodels")
    sys.exit(1)

from t1dsim_ai.individual_model import DigitalTwin

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
OUTPUT_MODEL_DIR = "output_models"
WINDOW_SIZE = 60   # 60 samples * 5 min = 5 hours, matches the paper
STEP = 60          # non-overlapping windows

MARGINS = {
    'TIR': 5.0, 'TAR': 5.0, 'TBR': 1.0, 'HBGI': 1.0, 'LBGI': 0.5, 'MG': 10.0,
}
PAPER_RMSE_BENCHMARK = {30: (27, 7), 60: (41, 12), 120: (54, 18)}
PAPER_TABLE3 = {  # for a side-by-side sanity check against the published numbers
    'TIR': (75.1, 21.2, 74.4, 15.4), 'TAR': (22.4, 22.0, 22.6, 15.9),
    'TBR': (2.5, 5.2, 3.0, 3.3),
}

PER_SUBJECT_OUTPUT_CSV = "digital_twin_cohort_results.csv"

# Try these candidate locations for each subject's preprocessed CSV, in order.
CSV_PATH_CANDIDATES = [
    lambda sid: os.path.join("data_processed", f"{sid}.csv"),
    lambda sid: f"D:\\data_processed\\{sid}.csv",
    lambda sid: f"src/t1dsim_ai/models/IndividualModel/{sid}/T1DEXIMAIN_{sid}.csv",
]


def resolve_csv_path(subject_id):
    for candidate_fn in CSV_PATH_CANDIDATES:
        path = candidate_fn(subject_id)
        if os.path.exists(path):
            return path
    return None


def get_subject_list():
    """Prefer an explicit list passed on the command line. Otherwise, fall
    back to every subject folder under output_models/ that has a trained
    model - this is exactly the set of subjects you already finished
    training, so no separate subject-list file is required."""
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1].strip())
        col = "subject_id" if "subject_id" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()

    subjects = []
    for d in sorted(glob.glob(os.path.join(OUTPUT_MODEL_DIR, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "individual_model.pt")):
            subjects.append(os.path.basename(d))
    return subjects


def compute_metrics(glucose):
    g = np.asarray(glucose, dtype=float)
    g = g[~np.isnan(g)]
    if len(g) == 0:
        return None
    tir = 100.0 * np.mean((g >= 70) & (g <= 180))
    tar = 100.0 * np.mean(g > 180)
    tbr = 100.0 * np.mean(g < 70)
    mg = np.mean(g)
    g_clipped = np.clip(g, 1, None)
    f_bg = 1.509 * (np.log(g_clipped) ** 1.084 - 5.381)
    lbgi = np.mean(np.where(f_bg < 0, 10 * f_bg ** 2, 0))
    hbgi = np.mean(np.where(f_bg > 0, 10 * f_bg ** 2, 0))
    return {'TIR': tir, 'TAR': tar, 'TBR': tbr, 'LBGI': lbgi, 'HBGI': hbgi, 'MG': mg}


def evaluate_one_subject(subject_id):
    """Returns (subject_mean_row_dict, list_of_actual_windows, list_of_sim_windows) or None if skipped."""
    csv_path = resolve_csv_path(subject_id)
    if csv_path is None:
        return None, "NO_CSV_FOUND"

    model_dir = os.path.abspath(os.path.join(OUTPUT_MODEL_DIR, subject_id))
    if not os.path.exists(os.path.join(model_dir, "individual_model.pt")):
        return None, "NO_MODEL_FOUND"

    df_data = pd.read_csv(csv_path)
    df_test = df_data[~df_data.is_train].reset_index(drop=True)
    n = len(df_test)
    if n < WINDOW_SIZE:
        return None, "TEST_SET_TOO_SHORT"

    dt = DigitalTwin(custom_DT=model_dir)

    actual_metrics_list, sim_metrics_list = [], []
    actual_windows, sim_windows = [], []
    for start in range(0, n - WINDOW_SIZE + 1, STEP):
        window_df = df_test.iloc[start:start + WINDOW_SIZE].reset_index(drop=True)
        try:
            df_sim_window = dt.simulate(window_df)
        except Exception:
            continue
        a = df_sim_window['cgm_Actual'].values
        s = df_sim_window['cgm_NNDT'].values
        m_a, m_s = compute_metrics(a), compute_metrics(s)
        if m_a is None or m_s is None:
            continue
        actual_metrics_list.append(m_a)
        sim_metrics_list.append(m_s)
        actual_windows.append(a)
        sim_windows.append(s)

    if not actual_metrics_list:
        return None, "NO_WINDOWS_SIMULATED"

    df_a = pd.DataFrame(actual_metrics_list)
    df_s = pd.DataFrame(sim_metrics_list)

    row = {"subjectID": subject_id, "n_windows": len(df_a)}
    for metric in MARGINS:
        row[f"{metric}_actual"] = df_a[metric].mean()
        row[f"{metric}_sim"] = df_s[metric].mean()

    return row, (actual_windows, sim_windows)


def run_tost_table3_style(df_results):
    print("\n" + "=" * 90)
    print("COHORT-WIDE PAIRED EQUIVALENCE TESTS (TOST) - paper's Table 3 methodology")
    print(f"(n = {len(df_results)} digital twins, one paired observation per subject)")
    print("=" * 90)
    header = f"{'Metric':<8}{'Simulated (mean±SD)':>24}{'Actual (mean±SD)':>22}{'margin':>9}{'p-value':>12}{'Equivalent?':>14}"
    print(header)
    for metric, margin in MARGINS.items():
        a = df_results[f"{metric}_actual"].values
        s = df_results[f"{metric}_sim"].values
        p_value, _, _ = ttost_paired(s, a, -margin, margin)
        equivalent = "YES" if p_value < 0.05 else "no"
        print(f"{metric:<8}{s.mean():>17.2f}±{s.std():<6.2f}{a.mean():>15.2f}±{a.std():<6.2f}"
              f"{margin:>9.2f}{p_value:>12.4f}{equivalent:>14}")
        if metric in PAPER_TABLE3:
            ps, psd, pa, pasd = PAPER_TABLE3[metric]
            print(f"         (paper Table 3: simulated {ps}±{psd}, actual {pa}±{pasd})")
    print("\n(p < 0.05 => equivalent within the clinical margin, matching the paper's criterion)")


def run_cohort_rmse(all_actual_windows, all_sim_windows, horizons_min=(30, 60, 120), step_min=5):
    """Paper Section 2.7.3: both the actual and simulated trajectories share
    the SAME t=0 initial condition for a given window. The h-minute-ahead
    "prediction" is simply the simulated value at index h within that
    window, compared to the actual value at that SAME index - NOT a shifted
    comparison between two different time points. One error sample per
    (window, horizon), matching "the predicted glucose value at time
    30/60/120 min was then compared to the actual value" (singular, per
    scenario) rather than a sliding-window comparison across the whole
    trace."""
    print("\n" + "=" * 90)
    print("COHORT-WIDE POINT-WISE PREDICTION RMSE (vs. paper's published benchmark)")
    print("=" * 90)
    for h in horizons_min:
        steps = h // step_min
        errors = []
        for a_win, s_win in zip(all_actual_windows, all_sim_windows):
            if steps >= len(a_win) or steps >= len(s_win):
                continue
            a_val = a_win[steps]
            s_val = s_win[steps]
            if np.isnan(a_val) or np.isnan(s_val):
                continue
            errors.append((a_val - s_val) ** 2)
        if not errors:
            continue
        rmse = np.sqrt(np.mean(errors))
        bm, bsd = PAPER_RMSE_BENCHMARK.get(h, (None, None))
        print(f"  {h}-min horizon: RMSE = {rmse:.1f} mg/dL  (n={len(errors)})  (paper: {bm}±{bsd} mg/dL)")


def main():
    subject_ids = get_subject_list()
    print(f"Found {len(subject_ids)} candidate subjects (trained models present).\n")

    rows = []
    all_actual_windows, all_sim_windows = [], []
    skipped = []

    for i, subject_id in enumerate(subject_ids, start=1):
        print(f"[{i}/{len(subject_ids)}] Evaluating {subject_id}...", end=" ")
        try:
            row, extra = evaluate_one_subject(subject_id)
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
            skipped.append({"subjectID": subject_id, "reason": f"EXCEPTION: {e}"})
            continue

        if row is None:
            print(f"skipped ({extra})")
            skipped.append({"subjectID": subject_id, "reason": extra})
            continue

        actual_windows, sim_windows = extra
        rows.append(row)
        all_actual_windows.extend(actual_windows)
        all_sim_windows.extend(sim_windows)
        print(f"ok ({row['n_windows']} windows)")

    df_results = pd.DataFrame(rows)
    df_results.to_csv(PER_SUBJECT_OUTPUT_CSV, index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv("cohort_eval_skipped.csv", index=False)

    print("\n" + "=" * 90)
    print(f" Successfully evaluated {len(df_results)} / {len(subject_ids)} subjects.")
    print(f" Per-subject mean outcomes saved to: {PER_SUBJECT_OUTPUT_CSV}")
    if skipped:
        print(f" {len(skipped)} subjects skipped - see cohort_eval_skipped.csv for reasons.")
    print("=" * 90)

    if len(df_results) == 0:
        print("Nothing to evaluate.")
        return

    run_tost_table3_style(df_results)
    run_cohort_rmse(all_actual_windows, all_sim_windows)


if __name__ == "__main__":
    main()