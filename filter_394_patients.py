"""
=============================================================================
  Filter T1DEXI Participants to Match the Paper's Inclusion Criteria
=============================================================================
Replicates the participant-filtering rules described in the paper
(Sections 2.1 and 2.4.1) BEFORE you run per-subject preprocessing, so you
only preprocess/train the subjects that actually qualify.

Two exclusion rules, applied exactly as described in the paper:

  1. THERAPY MODALITY (Section 2.1): "As MDI data was self-reported, to
     ensure correct model identification and validation, we limit our
     analysis to participants using either open loop insulin pumps or AID
     systems." -> Any subject whose DX.xpt device history includes
     DXTRT == "Multiple Daily Injections" is EXCLUDED.

  2. DATA COMPLETENESS (Section 2.4.1): "If for a given participant, there
     were no insulin or carbohydrates reported in either of the two
     subsets, the participant was excluded from the virtual population."
     -> The first TRAIN_DAYS days (anchored to each subject's first CGM
     reading, same anchor used in preprocessing) are the "train" subset,
     the remaining days are "test". A subject is EXCLUDED if either
     subset has zero non-zero insulin dose records OR zero non-zero carb
     records.

Output: a CSV (default: included_subjects.csv) listing every subject with
their pass/fail status per rule, plus a second CSV with just the final
included subject IDs (formatted as T1DEXI-01-XXXX), ready to feed into the
preprocessing / training steps.

USAGE:
    python filter_394_patients.py

Adjust RAW_DATA_DIR / FILE_PATH_OVERRIDES below to match your setup (same
convention as the Step 1 preprocessing script).
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION - match these to your Step 1 preprocessing script
# ---------------------------------------------------------------------------
RAW_DATA_DIR = r"D:\T1DEXI - DATA FOR UPLOAD\T1DEXI - DATA FOR UPLOAD"

FILE_PATH_OVERRIDES = {
    "FAMLPM.xpt": r"C:\Users\3com\OneDrive\Desktop\FAMLPM.xpt",
    # "FACM.xpt": r"...",  # add here too if FACM also had to be copied locally
}

TRAIN_DAYS = 14  # first two weeks = train, matching paper Section 2.4.1
MDI_LABEL = "Multiple Daily Injections"

OUT_FULL_REPORT = "participant_filter_report.csv"
OUT_INCLUDED_IDS = "included_subjects.csv"

CHUNKSIZE = 200_000


def resolve_path(filename):
    return FILE_PATH_OVERRIDES.get(filename, os.path.join(RAW_DATA_DIR, filename))


def decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


def clean_id(usubjid):
    """USUBJID -> the zero-stripped numeric id used elsewhere in the pipeline."""
    s = decode(usubjid)
    s = str(s).strip()
    stripped = s.lstrip("0")
    return stripped if stripped else s


# ---------------------------------------------------------------------------
# Step 1: full subject roster from DM.xpt
# ---------------------------------------------------------------------------
def get_all_subjects():
    path = resolve_path("DM.xpt")
    subjects = set()
    reader = pd.read_sas(path, format="xport", chunksize=CHUNKSIZE)
    for chunk in reader:
        ids = chunk["USUBJID"].apply(clean_id)
        subjects.update(ids.tolist())
    return subjects


# ---------------------------------------------------------------------------
# Step 2: each subject's first CGM reading (anchor for the 14-day cutoff)
# ---------------------------------------------------------------------------
def get_cgm_start_times():
    path = resolve_path("LB.xpt")
    starts = {}
    reader = pd.read_sas(
        path, format="xport", chunksize=CHUNKSIZE
    )
    for chunk in reader:
        cols = [c for c in ["USUBJID", "LBTESTCD", "LBDTC"] if c in chunk.columns]
        chunk = chunk[cols]
        chunk = chunk[chunk["LBTESTCD"].apply(decode) == "GLUC"]
        if len(chunk) == 0:
            continue
        chunk["sid"] = chunk["USUBJID"].apply(clean_id)
        chunk["dt"] = pd.to_datetime(chunk["LBDTC"], unit="s", errors="coerce")
        chunk = chunk.dropna(subset=["dt"])
        for sid, dt_min in chunk.groupby("sid")["dt"].min().items():
            if sid not in starts or dt_min < starts[sid]:
                starts[sid] = dt_min
    return starts


# ---------------------------------------------------------------------------
# Step 3: therapy modality per subject from DX.xpt
# ---------------------------------------------------------------------------
def get_device_sets():
    """Returns {sid: set(DXTRT values as they appear in the raw data)}.
    Matching against MDI_LABEL is done case-insensitively later, since this
    dataset stores DXTRT in ALL CAPS while the define.xml codelist shows it
    in Title Case."""
    path = resolve_path("DX.xpt")
    devices = {}
    reader = pd.read_sas(path, format="xport", chunksize=CHUNKSIZE)
    for chunk in reader:
        cols = [c for c in ["USUBJID", "DXTRT"] if c in chunk.columns]
        chunk = chunk[cols]
        chunk["sid"] = chunk["USUBJID"].apply(clean_id)
        chunk["trt"] = chunk["DXTRT"].apply(decode).astype(str).str.strip()
        for sid, trt_set in chunk.groupby("sid")["trt"].apply(set).items():
            devices.setdefault(sid, set()).update(trt_set)
    return devices


# ---------------------------------------------------------------------------
# Step 4: train/test non-zero coverage for insulin (FACM) or carbs (FAMLPM)
# ---------------------------------------------------------------------------
def get_split_nonzero_flags(filename, testcd_values, cgm_start):
    """Returns {sid: (has_nonzero_in_train, has_nonzero_in_test)}."""
    path = resolve_path(filename)
    flags = {}  # sid -> [has_train, has_test]
    reader = pd.read_sas(path, format="xport", chunksize=CHUNKSIZE)
    for chunk in reader:
        cols = [c for c in ["USUBJID", "FATESTCD", "FADTC", "FASTRESN"] if c in chunk.columns]
        chunk = chunk[cols]
        chunk = chunk[chunk["FATESTCD"].apply(decode).isin(testcd_values)]
        if len(chunk) == 0:
            continue
        chunk["sid"] = chunk["USUBJID"].apply(clean_id)
        chunk["dt"] = pd.to_datetime(chunk["FADTC"], unit="s", errors="coerce")
        chunk["val"] = pd.to_numeric(chunk["FASTRESN"], errors="coerce")
        chunk = chunk.dropna(subset=["dt", "val"])
        chunk = chunk[chunk["val"] != 0]
        if len(chunk) == 0:
            continue
        for sid, grp in chunk.groupby("sid"):
            start = cgm_start.get(sid)
            if start is None:
                continue
            cutoff = start + pd.Timedelta(days=TRAIN_DAYS)
            is_train = (grp["dt"] < cutoff).any()
            is_test = (grp["dt"] >= cutoff).any()
            cur = flags.setdefault(sid, [False, False])
            cur[0] = cur[0] or bool(is_train)
            cur[1] = cur[1] or bool(is_test)
    return {sid: tuple(v) for sid, v in flags.items()}


def main():
    print("=" * 70)
    print(" Filtering T1DEXI participants to match the paper's criteria")
    print("=" * 70)

    print("\n[1/5] Reading full subject roster (DM.xpt)...")
    subjects = get_all_subjects()
    print(f"      Total subjects found: {len(subjects)}")

    print("\n[2/5] Reading first CGM reading per subject (LB.xpt, GLUC)...")
    cgm_start = get_cgm_start_times()
    print(f"      Subjects with at least one CGM reading: {len(cgm_start)}")

    print("\n[3/5] Reading therapy/device history (DX.xpt)...")
    devices = get_device_sets()
    print(f"      Subjects with device records: {len(devices)}")

    print("\n[4/5] Checking insulin coverage per split (FACM.xpt: BASFLRT, INSBOLUS)...")
    insulin_flags = get_split_nonzero_flags("FACM.xpt", {"BASFLRT", "INSBOLUS"}, cgm_start)

    print("\n[5/5] Checking carb coverage per split (FAMLPM.xpt: DCARBT)...")
    carb_flags = get_split_nonzero_flags("FAMLPM.xpt", {"DCARBT"}, cgm_start)

    rows = []
    for sid in sorted(subjects, key=lambda x: (len(x), x)):
        dev = devices.get(sid, set())
        has_device_info = len(dev) > 0
        # Case-insensitive match: this dataset stores DXTRT in ALL CAPS.
        is_mdi = any(d.strip().upper() == MDI_LABEL.upper() for d in dev)

        has_cgm = sid in cgm_start
        ins_train, ins_test = insulin_flags.get(sid, (False, False))
        carb_train, carb_test = carb_flags.get(sid, (False, False))

        complete_train = ins_train and carb_train
        complete_test = ins_test and carb_test
        data_complete = complete_train and complete_test

        included = has_cgm and has_device_info and (not is_mdi) and data_complete

        rows.append({
            "usubjid_clean": sid,
            "subject_id": f"T1DEXI-01-{sid.zfill(4)}",
            "has_cgm": has_cgm,
            "device_names": "; ".join(sorted(dev)) if dev else "",
            "has_device_info": has_device_info,
            "is_mdi": is_mdi,
            "insulin_in_train": ins_train,
            "insulin_in_test": ins_test,
            "carbs_in_train": carb_train,
            "carbs_in_test": carb_test,
            "data_complete": data_complete,
            "INCLUDED": included,
        })

    report = pd.DataFrame(rows)
    report.to_csv(OUT_FULL_REPORT, index=False)

    included = report[report["INCLUDED"]]
    included[["subject_id"]].to_csv(OUT_INCLUDED_IDS, index=False)

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f" Total subjects checked:          {len(report)}")
    print(f" Excluded - no CGM data:          {(~report.has_cgm).sum()}")
    print(f" Excluded - no device info:       {(report.has_cgm & ~report.has_device_info).sum()}")
    print(f" Excluded - MDI:                  {(report.is_mdi).sum()}")
    print(f" Excluded - incomplete ins/carbs: {(report.has_device_info & ~report.is_mdi & ~report.data_complete).sum()}")
    print(f" INCLUDED (final cohort):         {included.shape[0]}")
    print(f"\n Full report saved to: {OUT_FULL_REPORT}")
    print(f" Included subject IDs saved to:  {OUT_INCLUDED_IDS}")
    print("=" * 70)


if __name__ == "__main__":
    main()