"""
=============================================================================
  Step 1 (FIXED v3 + BATCH): Raw Data Preprocessing for T1DEXI Dataset
=============================================================================
Same batch driver as before (loops over every subject in
included_subjects.csv). Only the HEART-RATE baseline / scaling changed in v3
(everything else is identical to v2):

  v3 change 1 - HR BASELINE = mean HR at REST (paper Section 2.4.1)
      The paper defines the baseline as "the average heart rate during
      periods of rest, when no exercise was being performed".
      Before: median of ALL HR (exercise included, whole recording).
      Now:    mean of the observed 5-min HR buckets that are
                (a) inside the TRAINING period only (first TRAIN_DAYS days), and
                (b) outside every logged exercise window.
      Exercise windows come from PR.xpt (define.xml: PRSTDTC = start of the
      reported activity, PLNEXDUR = reported activity duration in minutes),
      padded by EXERCISE_PAD_BEFORE_MIN before and EXERCISE_PAD_AFTER_MIN
      after (HR needs time to come back down).
      If a subject has no PR records (or too few rest buckets), a documented
      fallback is used and printed in the log.

  v3 change 2 - ROBUST SCALING of the HR deviation (paper, Appendix C)
      scaled = (x - median) / IQR  with x = heart_rate - baseline.
      median / IQR are computed from the TRAINING rows only and are applied
      unchanged to the test rows (no leakage). The constants are saved per
      subject to  <subject_id>_hr_scaler.json  so the model code can
      re-use them. New CSV column: heart_rate_WRTbaseline_scaled
      (heart_rate_WRTbaseline is still written, in raw BPM).

Inherited from v2:
  A. HEART RATE uses VSTESTCD == 'HRM', averaged into 5-min buckets, gaps up
     to HR_MAX_GAP_BUCKETS interpolated, longer gaps filled with the baseline.
  B. SLEEP EFFICIENCY is held over the whole [NVDTC, NVENDTC) interval.

USAGE:
    python preprocess_all_subjects.py
    python preprocess_all_subjects.py path/to/included_subjects.csv

To try it on a few subjects first, pass a small CSV with a 'subject_id'
column (e.g. T1DEXI-01-1004, T1DEXI-01-0692, T1DEXI-01-0011).
"""

import os
import re
import sys
import json
import traceback
import pandas as pd
import numpy as np

from meal_reconciliation import reconcile_meals, print_stats

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RAW_DATA_DIR = r"D:\T1DEXI - DATA FOR UPLOAD\T1DEXI - DATA FOR UPLOAD"

# If a domain file had to be copied elsewhere to read reliably, point to it
# here. Anything not listed is read from RAW_DATA_DIR as normal.
FILE_PATH_OVERRIDES = {
    "FAMLPM.xpt": r"C:\Users\3com\OneDrive\Desktop\FAMLPM.xpt",
    "VS.xpt": r"C:\Users\3com\OneDrive\Desktop\VS.xpt",
}

# CSV produced by the filtering script. Must contain a "subject_id" column
# formatted as T1DEXI-01-XXXX (same format written by filter_394_patients.py).
INCLUDED_SUBJECTS_CSV = "included_subjects.csv"
if len(sys.argv) > 1:
    INCLUDED_SUBJECTS_CSV = sys.argv[1].strip()

GRID_FREQ_MIN = 5
TRAIN_DAYS = 14  # first two weeks = train, matching paper Section 2.4.1

HR_TEST_CODE = "HRM"        # continuous 10-second wearable heart rate
HR_MAX_GAP_BUCKETS = 6      # interpolate HR gaps up to 6 x 5 min = 30 min
HR_DEFAULT_BASELINE = 64.0  # only used if a subject has no HR data at all

# --- exercise exclusion for the HR baseline (v3) ---------------------------
EXERCISE_PAD_BEFORE_MIN = 15        # HR can start rising before the logged start
EXERCISE_PAD_AFTER_MIN = 60         # recovery time after the logged end
DEFAULT_EXERCISE_DURATION_MIN = 60  # used when PLNEXDUR is missing / <= 0
MIN_REST_BUCKETS = 288              # need >= 1 day (288 x 5 min) of rest HR
MIN_SCALER_BUCKETS = 100            # min observed train buckets to fit scaler

BATCH_REPORT_CSV = "batch_run_report.csv"

# Output folder for the preprocessed per-subject CSVs - saved on D: instead
# of wherever the script happens to be launched from.
DATA_PROCESSED_DIR = r"D:\data_processed"
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)


def resolve_path(filename):
    return FILE_PATH_OVERRIDES.get(filename, os.path.join(RAW_DATA_DIR, filename))


def decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


def decode_cols(df, cols):
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].apply(decode)
    return df


def read_full_for_subject(filename, target_clean, keep_cols=None):
    """Read a small-to-medium .xpt file, filter to one subject."""
    path = resolve_path(filename)
    if not os.path.exists(path):
        print(f"      [Warning] {filename} not found at {path}")
        return pd.DataFrame()
    reader = pd.read_sas(path, format="xport", chunksize=200000)
    matched = []
    for chunk in reader:
        if keep_cols is not None:
            available = [c for c in keep_cols if c in chunk.columns]
            chunk = chunk[available]
        sub_col = chunk['USUBJID'].apply(decode).astype(str).str.lstrip('0')
        hit = chunk[sub_col == target_clean]
        if len(hit):
            matched.append(hit)
    if not matched:
        return pd.DataFrame()
    df = pd.concat(matched, ignore_index=True)
    return decode_cols(df, [c for c in df.columns if df[c].dtype == object])


def parse_iso8601_duration_to_minutes(s):
    """Parse things like 'P0Y0M0DT4H0M0S' -> minutes (float)."""
    if not s or not isinstance(s, str):
        return np.nan
    m = re.match(
        r"P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?",
        s
    )
    if not m:
        return np.nan
    y, mo, d, h, mi, sec = [float(x) if x else 0.0 for x in m.groups()]
    total_minutes = y * 525600 + mo * 43800 + d * 1440 + h * 60 + mi + sec / 60.0
    return total_minutes


def build_time_grid(start_time, end_time, freq_min=GRID_FREQ_MIN):
    start = start_time.floor(f"{freq_min}min")
    end = end_time.ceil(f"{freq_min}min")
    return pd.date_range(start=start, end=end, freq=f"{freq_min}min")


def bucketize_point_events(df_events, time_col, value_col, grid, freq_min=GRID_FREQ_MIN):
    """Round each event's timestamp DOWN to its grid bucket, sum values
    landing in the same bucket, and reindex onto the full grid (missing ->
    0). Used for bolus doses and meal carbs (discrete point events)."""
    out = pd.Series(0.0, index=grid)
    if len(df_events) == 0:
        return out
    df = df_events.copy()
    df['bucket'] = df[time_col].dt.floor(f"{freq_min}min")
    summed = df.groupby('bucket')[value_col].sum()
    out.loc[out.index.intersection(summed.index)] = summed.loc[out.index.intersection(summed.index)]
    return out


def build_basal_dose_series(df_basal, grid, freq_min=GRID_FREQ_MIN):
    """df_basal has FADTC (start time) and FASTRESN (rate, U/hr). Build a
    step function of the active rate (forward-filled from each change
    point). Returns the instantaneous U/hr rate at each grid point
    (NOT the dose-per-bucket) so it matches the scale the ODE Pop model
    expects for input_insulin."""
    out = pd.Series(0.0, index=grid)
    if len(df_basal) == 0:
        return out
    df = df_basal.sort_values('FADTC')[['FADTC', 'FASTRESN']].dropna()
    if len(df) == 0:
        return out
    # as-of merge: for each grid timestamp, find the most recent rate change
    rate_series = pd.merge_asof(
        pd.DataFrame({'grid_time': grid}),
        df.rename(columns={'FADTC': 'grid_time', 'FASTRESN': 'rate_u_per_hr'}),
        on='grid_time', direction='backward'
    )
    rate_series['rate_u_per_hr'] = rate_series['rate_u_per_hr'].fillna(0.0)
    # Return the rate directly (U/hr), not the dose (rate * freq_min/60)
    out[:] = rate_series['rate_u_per_hr'].values
    return out


# ---------------------------------------------------------------------------
# Heart rate helpers (v3)
# ---------------------------------------------------------------------------
def parse_sdtm_datetime(s):
    """Parse an SDTM date/time column.
    - numeric column  -> epoch seconds (same convention as VSDTC / LBDTC / FADTC)
    - text column     -> ISO-8601 string (define.xml lists PRSTDTC as *text*)
    Result is always tz-naive. NOTE: ISO strings that carry a UTC offset are
    converted to UTC first; if you see exercise windows shifted relative to
    the HR curve, check this (the log prints the first PR start time)."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit='s', errors='coerce')
    try:
        out = pd.to_datetime(s, errors='coerce', utc=True, format='ISO8601')   # pandas >= 2.0
    except (ValueError, TypeError):
        out = pd.to_datetime(s, errors='coerce', utc=True)                      # older pandas
    return out.dt.tz_localize(None)


def load_exercise_intervals(target_clean):
    """Exercise sessions for one subject from PR.xpt.
    PRSTDTC = start, PLNEXDUR = reported duration in minutes (define.xml).
    Returns a DataFrame with columns ['start', 'end'] (possibly empty)."""
    empty = pd.DataFrame({'start': pd.to_datetime([]), 'end': pd.to_datetime([])})
    sub_pr = read_full_for_subject(
        "PR.xpt", target_clean,
        keep_cols=['USUBJID', 'PRTRT', 'PRCAT', 'PRSTDTC', 'PLNEXDUR']
    )
    if len(sub_pr) == 0 or 'PRSTDTC' not in sub_pr.columns:
        return empty
    start = parse_sdtm_datetime(sub_pr['PRSTDTC'])
    if 'PLNEXDUR' in sub_pr.columns:
        dur = pd.to_numeric(sub_pr['PLNEXDUR'], errors='coerce')
    else:
        dur = pd.Series(np.nan, index=sub_pr.index)
    dur = dur.where(dur > 0).fillna(DEFAULT_EXERCISE_DURATION_MIN)
    iv = pd.DataFrame({'start': start, 'end': start + pd.to_timedelta(dur, unit='m')})
    iv = iv.dropna(subset=['start']).sort_values('start').reset_index(drop=True)
    return iv


def build_exercise_mask(df_exercise, grid):
    """Boolean Series on the grid: True inside any exercise window
    [start - PAD_BEFORE, end + PAD_AFTER]."""
    mask = np.zeros(len(grid), dtype=bool)
    for _, r in df_exercise.iterrows():
        s = r['start'] - pd.Timedelta(minutes=EXERCISE_PAD_BEFORE_MIN)
        e = r['end'] + pd.Timedelta(minutes=EXERCISE_PAD_AFTER_MIN)
        i0 = grid.searchsorted(s, side="left")
        i1 = grid.searchsorted(e, side="right")
        mask[i0:i1] = True
    return pd.Series(mask, index=grid)


def bucket_hr(df_hr, grid, freq_min=GRID_FREQ_MIN):
    """5-minute mean of the raw HR samples on the grid. NaN where the
    wearable produced no samples (nothing is filled here)."""
    if df_hr is None or len(df_hr) == 0:
        return pd.Series(np.nan, index=grid)
    bucket_mean = df_hr.groupby(df_hr['datetime'].dt.floor(f"{freq_min}min"))['heart_rate'].mean()
    return bucket_mean.reindex(grid)


def compute_hr_baseline(hr_raw, exercise_mask, train_mask, has_exercise_log,
                        default_baseline=HR_DEFAULT_BASELINE):
    """Baseline = mean HR at rest (paper): observed buckets, TRAIN period only,
    outside exercise windows. Returns (baseline, method_description)."""
    observed_train = hr_raw.notna() & train_mask
    if not observed_train.any():
        return default_baseline, "default (no HR in training period)"
    if has_exercise_log:
        rest = observed_train & ~exercise_mask
        if int(rest.sum()) >= MIN_REST_BUCKETS:
            return float(hr_raw[rest].mean()), \
                f"rest mean ({int(rest.sum())} observed buckets outside exercise)"
    # Fallback: no exercise log, or too little rest data. Approximate "rest"
    # as the lower half of the observed training HR distribution.
    vals = hr_raw[observed_train]
    low = vals[vals <= vals.median()]
    why = "no PR exercise records" if not has_exercise_log else "too few rest buckets"
    return float(low.mean()), f"FALLBACK mean of HR <= median ({why})"


def fill_hr_series(hr_raw, baseline, max_gap_buckets=HR_MAX_GAP_BUCKETS):
    """Gaps <= max_gap_buckets are linearly interpolated (in time); longer
    gaps / edges are filled with the baseline ("no deviation")."""
    is_na = hr_raw.isna()
    if not is_na.any():
        return hr_raw.copy()
    run_id = (is_na != is_na.shift()).cumsum()
    run_len = is_na.groupby(run_id).transform("sum")   # length of each NaN run
    interpolated = hr_raw.interpolate(method="time", limit_area="inside")
    too_long = is_na & (run_len > max_gap_buckets)
    return interpolated.mask(too_long).fillna(baseline)


def fit_robust_scaler(x):
    """Robust scaler constants: center = median, scale = IQR (Q75 - Q25).
    Falls back to the std (then 1.0) if the IQR is ~0, e.g. when most values
    are exactly 0 because of baseline-filled gaps."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    center = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    scale = float(q75 - q25)
    if scale < 1e-6:
        std = float(np.std(x))
        scale = std if std > 1e-6 else 1.0
    return center, scale


def build_sleep_series(df_sleep, grid, freq_min=GRID_FREQ_MIN):
    """df_sleep has 'start', 'end' (may be NaT) and 'se_value'. The value is
    held over the whole [start, end) interval. If end is missing/invalid,
    only the single bucket containing start is filled."""
    values = np.zeros(len(grid))
    for _, r in df_sleep.iterrows():
        start = r["start"].floor(f"{freq_min}min")
        if pd.notna(r["end"]) and r["end"] > r["start"]:
            end = r["end"]
        else:
            end = start + pd.Timedelta(minutes=freq_min)
        i0 = grid.searchsorted(start, side="left")
        i1 = grid.searchsorted(end, side="left")
        values[i0:i1] = r["se_value"]
    return pd.Series(values, index=grid)


# ---------------------------------------------------------------------------
# Helpers to load the list of subjects to process
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
            f"Could not find {csv_path}. Run filter_394_patients.py first, "
            f"or pass the correct path as a command-line argument."
        )
    df = pd.read_csv(csv_path)
    if "subject_id" not in df.columns:
        raise ValueError(
            f"'{csv_path}' has no 'subject_id' column. Expected the format "
            f"produced by filter_394_patients.py (included_subjects.csv)."
        )
    subject_ids = df["subject_id"].astype(str).tolist()
    return subject_ids


# ---------------------------------------------------------------------------
# Core per-subject preprocessing
# ---------------------------------------------------------------------------
def preprocess_subject(target_usubjid):
    clean_target = target_usubjid.lstrip('0') if target_usubjid.lstrip('0') else target_usubjid
    subject_id = f"T1DEXI-01-{clean_target.zfill(4)}"
    output_csv = os.path.join(DATA_PROCESSED_DIR, f"{subject_id}.csv")
    scaler_json = os.path.join(DATA_PROCESSED_DIR, f"{subject_id}_hr_scaler.json")

    print("=" * 70)
    print(f" Preprocessing Subject {subject_id} (USUBJID: {clean_target})")
    print("=" * 70)

    # 1. Demographics -------------------------------------------------------
    print("\n[1/6] Loading Demographics (DM.xpt)...")
    sub_dm = read_full_for_subject("DM.xpt", clean_target)
    if len(sub_dm) == 0:
        raise ValueError(f"Subject {target_usubjid} not found in DM.xpt!")
    print(f"      Found subject. Age={sub_dm['AGE'].values[0]}, Sex={sub_dm['SEX'].values[0]}")

    # 2. CGM glucose ----------------------------------------------------------
    print("\n[2/6] Loading Glucose Measurements (LB.xpt)...")
    sub_lb = read_full_for_subject(
        "LB.xpt", clean_target,
        keep_cols=['USUBJID', 'LBTESTCD', 'LBDTC', 'LBSTRESN']
    )
    sub_lb = sub_lb[sub_lb['LBTESTCD'] == 'GLUC'].copy()
    if len(sub_lb) == 0:
        raise ValueError(f"No CGM glucose readings found for subject {clean_target}!")
    sub_lb['datetime'] = pd.to_datetime(sub_lb['LBDTC'], unit='s', errors='coerce')
    sub_lb['cgm'] = pd.to_numeric(sub_lb['LBSTRESN'], errors='coerce')
    sub_lb = sub_lb.sort_values('datetime').dropna(subset=['datetime', 'cgm'])
    print(f"      Found {len(sub_lb)} glucose readings.")

    # 3. Insulin: basal (rate) + bolus (dose), split correctly ---------------
    print("\n[3/6] Loading Insulin (FACM.xpt) - splitting basal rate vs bolus dose...")
    sub_facm = read_full_for_subject(
        "FACM.xpt", clean_target,
        keep_cols=['USUBJID', 'FATESTCD', 'FADTC', 'FASTRESN']
    )
    if len(sub_facm):
        sub_facm['FADTC'] = pd.to_datetime(sub_facm['FADTC'], unit='s', errors='coerce')
        sub_facm['FASTRESN'] = pd.to_numeric(sub_facm['FASTRESN'], errors='coerce')
        df_basal = sub_facm[sub_facm['FATESTCD'] == 'BASFLRT'].dropna(subset=['FADTC'])
        df_bolus = sub_facm[sub_facm['FATESTCD'] == 'INSBOLUS'].dropna(subset=['FADTC', 'FASTRESN'])
        print(f"      Basal rate-change events: {len(df_basal)} | Bolus doses: {len(df_bolus)}")
    else:
        print("      No FACM records found, insulin will be all zero.")
        df_basal, df_bolus = pd.DataFrame(), pd.DataFrame()

    # 4. Carbs from FAMLPM (DCARBT - Dietary Total Carbohydrates in grams) ----
    #    We extract DCARBT and prioritize FAMETHOD == 'SELF-REPORT' (the patient's
    #    own carb count for the meal, which matches what they dosed insulin for).
    #    If a timestamp only has RFPM CONSUMED, we use that as fallback.
    #    This avoids food portion weights from ML.xpt and avoids summing duplicate
    #    records (TAKEN / RETURNED / CONSUMED / SELF-REPORT) at the same timestamp.
    print("\n[4/6] Loading Meal Carbohydrates (FAMLPM.xpt, FATESTCD == 'DCARBT')...")
    sub_famlpm = read_full_for_subject(
        "FAMLPM.xpt", clean_target,
        keep_cols=['USUBJID', 'FATESTCD', 'FAMETHOD', 'FACAT', 'FADTC', 'FASTRESN']
    )
    if len(sub_famlpm):
        sub_famlpm['FATESTCD'] = sub_famlpm['FATESTCD'].apply(decode)
        sub_famlpm['FAMETHOD'] = sub_famlpm['FAMETHOD'].apply(decode)
        sub_famlpm['FACAT'] = sub_famlpm['FACAT'].apply(decode)

        dcarb = sub_famlpm[sub_famlpm['FATESTCD'] == 'DCARBT'].copy()
        dcarb['FADTC'] = pd.to_datetime(dcarb['FADTC'], unit='s', errors='coerce')
        dcarb['FASTRESN'] = pd.to_numeric(dcarb['FASTRESN'], errors='coerce')
        dcarb = dcarb.dropna(subset=['FADTC', 'FASTRESN'])

        # 1. Primary: SELF-REPORT records
        sr = dcarb[dcarb['FAMETHOD'] == 'SELF-REPORT'].drop_duplicates(subset=['FADTC'])

        # 2. Fallback: RFPM CONSUMED records (only for timestamps not already covered by SELF-REPORT)
        rfpm_consumed = dcarb[(dcarb['FAMETHOD'] == 'RFPM') & (dcarb['FACAT'] == 'CONSUMED')]
        rfpm_fallback = rfpm_consumed[~rfpm_consumed['FADTC'].isin(sr['FADTC'])].drop_duplicates(subset=['FADTC'])

        df_carbs = pd.concat([sr, rfpm_fallback], ignore_index=True).sort_values('FADTC')
        print(f"      Meal carb records: {len(df_carbs)} (SELF-REPORT: {len(sr)}, RFPM fallback: {len(rfpm_fallback)})")
        if len(df_carbs):
            print(f"      Carb amounts range: {df_carbs['FASTRESN'].min():.1f} - {df_carbs['FASTRESN'].max():.1f} g "
                  f"(median: {df_carbs['FASTRESN'].median():.1f} g)")
    else:
        print("      No FAMLPM records found, carbs will be all zero.")
        df_carbs = pd.DataFrame()

    # 5. Heart rate from VS: continuous HRM samples ---------------------------
    print(f"\n[5/6] Loading Heart Rate (VS.xpt, VSTESTCD == '{HR_TEST_CODE}')...")
    sub_vs = read_full_for_subject(
        "VS.xpt", clean_target,
        keep_cols=['USUBJID', 'VSTESTCD', 'VSDTC', 'VSSTRESN']
    )
    if len(sub_vs):
        sub_vs = sub_vs[sub_vs['VSTESTCD'] == HR_TEST_CODE].copy()
        sub_vs['datetime'] = pd.to_datetime(sub_vs['VSDTC'], unit='s', errors='coerce')
        sub_vs['heart_rate'] = pd.to_numeric(sub_vs['VSSTRESN'], errors='coerce')
        sub_vs = sub_vs.dropna(subset=['datetime', 'heart_rate'])
        if len(sub_vs):
            print(f"      Found {len(sub_vs)} {HR_TEST_CODE} samples "
                  f"(HR range {sub_vs['heart_rate'].min():.0f}-{sub_vs['heart_rate'].max():.0f} BPM).")
        else:
            print(f"      No usable {HR_TEST_CODE} samples, heart rate will use a flat default.")
    else:
        sub_vs = pd.DataFrame()
        print("      No VS records found, heart rate will use a flat default.")

    # 5b. Exercise sessions from PR (used ONLY to exclude exercise from the
    #     resting HR baseline) --------------------------------------------------
    print("      Loading exercise sessions (PR.xpt: PRSTDTC + PLNEXDUR)...")
    df_exercise = load_exercise_intervals(clean_target)
    if len(df_exercise):
        print(f"      Found {len(df_exercise)} exercise sessions | "
              f"first starts at {df_exercise['start'].iloc[0]} "
              f"(check this lines up with the HR / CGM time range)")
    else:
        print("      No exercise sessions found in PR.xpt for this subject.")

    # 6. Sleep efficiency from NV: value held over [NVDTC, NVENDTC) ------------
    print("\n[6/6] Loading Sleep Efficiency (NV.xpt, NVTESTCD == 'SE')...")
    sub_nv = read_full_for_subject(
        "NV.xpt", clean_target,
        keep_cols=['USUBJID', 'NVTESTCD', 'NVDTC', 'NVENDTC', 'NVSTRESN']
    )
    df_sleep = pd.DataFrame()
    if len(sub_nv):
        sub_nv = sub_nv[sub_nv['NVTESTCD'] == 'SE'].copy()
        sub_nv['start'] = pd.to_datetime(sub_nv['NVDTC'], unit='s', errors='coerce')
        sub_nv['end'] = pd.to_datetime(sub_nv['NVENDTC'], unit='s', errors='coerce') \
            if 'NVENDTC' in sub_nv.columns else pd.NaT
        sub_nv['se_value'] = pd.to_numeric(sub_nv['NVSTRESN'], errors='coerce')
        df_sleep = sub_nv.dropna(subset=['start', 'se_value']).copy()
        # Normalize to a 0-1 fraction if it looks like a 0-100 percentage
        if len(df_sleep) and df_sleep['se_value'].max() > 1.5:
            df_sleep['se_value'] = df_sleep['se_value'] / 100.0
        if len(df_sleep):
            dur = (df_sleep['end'] - df_sleep['start']).dt.total_seconds() / 60.0
            print(f"      Found {len(df_sleep)} sleep records | "
                  f"median interval {dur.median():.0f} min | "
                  f"value range {df_sleep['se_value'].min():.2f}-{df_sleep['se_value'].max():.2f}")
        else:
            print("      NV had no usable SE records.")
    else:
        print("      No NV sleep records found, sleep_efficiency will default to 0.")

    # -------------------------------------------------------------------
    # Build the 5-minute grid and assemble all signals onto it
    # -------------------------------------------------------------------
    print("\nBuilding 5-minute grid and merging all signals...")
    start_time = sub_lb['datetime'].min()
    end_time = sub_lb['datetime'].max()
    grid = build_time_grid(start_time, end_time)
    df_grid = pd.DataFrame({'datetime_local': grid})

    # CGM
    df_grid = pd.merge_asof(
        df_grid.sort_values('datetime_local'),
        sub_lb[['datetime', 'cgm']].sort_values('datetime'),
        left_on='datetime_local', right_on='datetime',
        direction='nearest', tolerance=pd.Timedelta('2.5min')
    ).drop(columns=['datetime'])
    df_grid['output_cgm'] = df_grid['cgm'].interpolate(method='linear', limit=3)

    # Train/test cutoff: first two calendar weeks = train (paper Section 2.4.1).
    # Computed here (before the HR features) because the HR baseline and the
    # HR scaler must be estimated from TRAINING data only. It is the same
    # value the old code computed after the dropna below.
    first_valid_time = df_grid.loc[df_grid['output_cgm'].notna(), 'datetime_local'].min()
    train_cutoff = first_valid_time + pd.Timedelta(days=TRAIN_DAYS)

    # Insulin: basal rate (U/hr step-function) + bolus converted to U/hr
    #   Basal:  build_basal_dose_series already returns U/hr (not dose-per-bucket)
    #   Bolus:  FASTRESN is the bolus in Units; multiply by 12 (= 60 min / 5 min)
    #           to express it as the equivalent instantaneous U/hr rate for that
    #           5-min bucket, matching the scale the Population Model expects.
    basal_series = build_basal_dose_series(df_basal, grid)
    bolus_series = bucketize_point_events(df_bolus, 'FADTC', 'FASTRESN', grid)
    df_grid['input_insulin'] = basal_series.values + bolus_series.values * 12.0

    # Carbs (v4): paper Sec. 2.4.1 style reconciliation of reported meals with
    # boluses (90-min window, 1:1 pairing, bolus timestamp, missed-meal recovery,
    # merging of per-food-item records). See meal_reconciliation.py.
    df_meals, meal_stats = reconcile_meals(
        df_carbs, df_bolus,
        df_cgm=sub_lb[['datetime', 'cgm']],
        train_cutoff=train_cutoff,
    )
    print_stats(meal_stats)
    carbs_series = bucketize_point_events(df_meals, 'FADTC', 'FASTRESN', grid)
    df_grid['input_meal_carbs'] = carbs_series.values

    # Heart rate -----------------------------------------------------------
    #  1) 5-min mean of HRM samples (NaN where the watch gave nothing)
    #  2) baseline = mean resting HR: train period only, outside exercise
    #  3) short gaps interpolated, long gaps = baseline ("no deviation")
    hr_raw = bucket_hr(sub_vs, grid)
    exercise_mask = build_exercise_mask(df_exercise, grid)
    train_mask = pd.Series(grid < train_cutoff, index=grid)
    hr_baseline, baseline_method = compute_hr_baseline(
        hr_raw, exercise_mask, train_mask, has_exercise_log=len(df_exercise) > 0
    )
    hr_series = fill_hr_series(hr_raw, hr_baseline)
    hr_observed = hr_raw.notna()
    df_grid['heart_rate'] = hr_series.values
    df_grid['heart_rate_WRTbaseline'] = df_grid['heart_rate'] - hr_baseline   # raw BPM
    df_grid['hr_observed'] = hr_observed.values   # temporary, for the log only

    # Sleep efficiency (held over the whole sleep interval)
    if len(df_sleep):
        df_grid['sleep_efficiency'] = build_sleep_series(df_sleep, grid).values
    else:
        df_grid['sleep_efficiency'] = 0.0

    # Time-based contextual features
    hours = df_grid['datetime_local'].dt.hour + df_grid['datetime_local'].dt.minute / 60.0
    df_grid['feat_hour_of_day_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df_grid['feat_hour_of_day_cos'] = np.cos(2 * np.pi * hours / 24.0)
    df_grid['feat_is_weekend'] = (df_grid['datetime_local'].dt.weekday >= 5).astype(int)

    # Drop rows with no CGM ground truth, format time column
    df_grid = df_grid.dropna(subset=['output_cgm']).reset_index(drop=True)
    df_grid['time'] = df_grid['datetime_local'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Train/test split (cutoff computed above)
    df_grid['is_train'] = df_grid['datetime_local'] < train_cutoff

    # Robust scaling of the HR deviation: constants from TRAIN rows only,
    # then applied unchanged to every row (train + test). Observed buckets
    # are preferred so baseline-filled gaps don't collapse the IQR.
    fit_mask = df_grid['is_train'] & df_grid['hr_observed']
    if int(fit_mask.sum()) < MIN_SCALER_BUCKETS:
        fit_mask = df_grid['is_train']
    hr_center, hr_scale = fit_robust_scaler(df_grid.loc[fit_mask, 'heart_rate_WRTbaseline'])
    df_grid['heart_rate_WRTbaseline_scaled'] = \
        (df_grid['heart_rate_WRTbaseline'] - hr_center) / hr_scale

    with open(scaler_json, "w") as f:
        json.dump({
            "subject_id": subject_id,
            "hr_baseline_bpm": hr_baseline,
            "hr_baseline_method": baseline_method,
            "scaler_center_median": hr_center,
            "scaler_scale_iqr": hr_scale,
            "scaler_fit_rows": int(fit_mask.sum()),
            "train_cutoff": str(train_cutoff),
            "exercise_sessions": int(len(df_exercise)),
            "meal_reconciliation": {k: (None if (isinstance(v, float) and v != v) else v) for k, v in meal_stats.items()},
        }, f, indent=2)

    hr_cov = 100.0 * float(df_grid['hr_observed'].mean())
    sleep_cov = 100.0 * float((df_grid['sleep_efficiency'] > 0).mean())

    # Reorder columns to match the expected schema
    final_cols = ['time', 'datetime_local', 'heart_rate', 'output_cgm', 'input_insulin',
                  'input_meal_carbs', 'heart_rate_WRTbaseline', 'heart_rate_WRTbaseline_scaled',
                  'feat_is_weekend', 'feat_hour_of_day_sin', 'feat_hour_of_day_cos',
                  'sleep_efficiency', 'is_train']
    df_out = df_grid[final_cols]

    df_out.to_csv(output_csv, index=False)
    n_train = int(df_out['is_train'].sum())
    n_test = int((~df_out['is_train']).sum())
    print(f" SUCCESS! Preprocessed data saved to: {output_csv}")
    print(f" Total Samples: {len(df_out)} (Train: {n_train}, Test: {n_test})")
    print(f" HR coverage (buckets with real samples): {hr_cov:.1f}%")
    print(f" HR baseline: {hr_baseline:.1f} BPM  [{baseline_method}]")
    print(f" HR robust scaler (train only): center={hr_center:.2f} BPM, IQR={hr_scale:.2f} BPM "
          f"-> saved to {scaler_json}")
    print(f" Sleep coverage (buckets with sleep > 0): {sleep_cov:.1f}%")
    return output_csv, len(df_out), n_train, n_test, hr_cov, sleep_cov


# ---------------------------------------------------------------------------
# Batch driver: loop over every subject in included_subjects.csv
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f" Loading subject list from: {INCLUDED_SUBJECTS_CSV}")
    print("=" * 70)
    subject_ids = load_subject_list(INCLUDED_SUBJECTS_CSV)
    # أضف أو احذف أرقام المرضى هنا للتجربة
    subject_ids = ["T1DEXI-01-0011", "T1DEXI-01-0043", "T1DEXI-01-1004"]  # مثال: معالجة ثلاثة مرضى فقط
    print(f" Found {len(subject_ids)} subjects to preprocess.\n")

    results = []
    for i, subject_id in enumerate(subject_ids, start=1):
        target_usubjid = clean_id_from_subject_id(subject_id)
        print(f"\n>>> [{i}/{len(subject_ids)}] Subject {subject_id} (raw id: {target_usubjid})")
        try:
            output_csv, n_total, n_train, n_test, hr_cov, sleep_cov = preprocess_subject(target_usubjid)
            results.append({
                "subject_id": subject_id,
                "status": "OK",
                "output_csv": output_csv,
                "n_total": n_total,
                "n_train": n_train,
                "n_test": n_test,
                "hr_coverage_pct": round(hr_cov, 1),
                "sleep_coverage_pct": round(sleep_cov, 1),
                "error": "",
            })
        except Exception as e:
            print(f"      [FAILED] {subject_id}: {e}")
            results.append({
                "subject_id": subject_id,
                "status": "FAILED",
                "output_csv": "",
                "n_total": 0,
                "n_train": 0,
                "n_test": 0,
                "hr_coverage_pct": "",
                "sleep_coverage_pct": "",
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
            })
            continue

    report = pd.DataFrame(results)
    report.to_csv(BATCH_REPORT_CSV, index=False)

    n_ok = (report["status"] == "OK").sum()
    n_failed = (report["status"] == "FAILED").sum()
    print("\n" + "=" * 70)
    print(" BATCH SUMMARY")
    print("=" * 70)
    print(f" Total subjects attempted: {len(report)}")
    print(f" Succeeded:                {n_ok}")
    print(f" Failed:                   {n_failed}")
    print(f" Per-subject report saved to: {BATCH_REPORT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
