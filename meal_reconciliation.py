"""
meal_reconciliation.py
----------------------
Approximation of the meal-event processing in the T1DSim_AI paper (Sec. 2.4.1),
which relies on the meal detection / size estimation NN of Mosquera-Lopez et al.
(npj Digit Med 2023, ref [45]).

The NN itself is NOT reproduced (its 32 input features are in the paper's
Supplementary Table 1, and it was trained on UVA-Padova/OHSU simulations).
What IS reproduced is the paper's event-reconciliation logic, with a
rule-based substitute for the NN's outputs:

  paper                                       | here
  --------------------------------------------+-------------------------------
  meal + bolus within 90 min:                 | pair them 1:1 (closest first);
    bolus timestamp, NN-estimated carbs       |   time = bolus time,
                                              |   carbs = reported, but replaced
                                              |   by bolus*ICR if inconsistent
  reported meal, no bolus within 90 min:      | kept, reported time/carbs
    (reported carbs, detected timestamp)      |
  bolus with no reported meal within 90 min:  | kept as a "missed meal" only if
    (NN detects meal + estimates size)        |   CGM rises >= MIN_RISE_MGDL
                                              |   afterwards; carbs = bolus*ICR
  no meal counted twice                       | 1:1 pairing
  unreliable events discarded                 | tiny boluses / no CGM rise

Not reproduced: detection of meals that have NEITHER a report NEITHER a bolus,
and the NN's 20 g size classes. Call these limitations in the write-up.
"""
import numpy as np
import pandas as pd

WINDOW_MIN = 90            # paper: 90-minute matching window
MERGE_GAP_MIN = 15         # merge per-food-item records into one meal
DEFAULT_ICR = 10.0         # g/U, used only if too few pairs to estimate it
ICR_RANGE = (3.0, 60.0)    # sanity clip for the estimated ICR
MIN_PAIRS_FOR_ICR = 10
MAX_MEAL_G = 150.0         # cap for a single meal (log shows how many hit it)
MIN_MEAL_G = 5.0
MIN_BOLUS_U = 1.0          # smaller boluses are not used to infer meals
HYBRID_RATIO = 2.0         # reported vs bolus*ICR disagreement factor
MIN_RISE_MGDL = 20.0       # CGM rise needed to accept a bolus-only meal
RISE_HORIZON_MIN = 90


def _empty():
    return pd.DataFrame({
        "FADTC": pd.to_datetime([]),
        "FASTRESN": pd.Series([], dtype=float),
        "source": pd.Series([], dtype=object),
    })


def merge_close_records(meals, gap_min=MERGE_GAP_MIN):
    """Chain records closer than gap_min into one meal (sum carbs, first time).
    Fixes per-food-item RFPM rows being counted as separate/duplicate meals."""
    if len(meals) == 0:
        return meals
    m = meals.sort_values("FADTC").reset_index(drop=True)
    gid = m["FADTC"].diff().gt(pd.Timedelta(minutes=gap_min)).cumsum()
    return (m.groupby(gid)
              .agg(FADTC=("FADTC", "first"), FASTRESN=("FASTRESN", "sum"))
              .reset_index(drop=True))


def _pair_one_to_one(meal_t, bolus_t, window_min):
    """Greedy closest-first 1:1 matching. Returns {meal_idx: bolus_idx}."""
    cand = []
    for i, t in enumerate(meal_t):
        d = np.abs((bolus_t - t) / np.timedelta64(1, "m"))
        for j in np.where(d <= window_min)[0]:
            cand.append((float(d[j]), i, int(j)))
    cand.sort()
    used_m, used_b, out = set(), set(), {}
    for _, i, j in cand:
        if i in used_m or j in used_b:
            continue
        used_m.add(i)
        used_b.add(j)
        out[i] = j
    return out


def _cgm_rise(cgm_t, cgm_v, t, horizon_min=RISE_HORIZON_MIN):
    """Max CGM rise above the value at t over the next horizon (mg/dL)."""
    i0 = np.searchsorted(cgm_t, np.datetime64(t), side="left")
    i1 = np.searchsorted(cgm_t, np.datetime64(t + pd.Timedelta(minutes=horizon_min)),
                         side="right")
    if i0 >= len(cgm_t) or i1 - i0 < 3:
        return np.nan
    seg = cgm_v[i0:i1]
    return float(np.nanmax(seg) - seg[0])


def reconcile_meals(df_carbs, df_bolus, df_cgm=None, train_cutoff=None,
                    window_min=WINDOW_MIN, size_source="hybrid"):
    """
    df_carbs : DataFrame[FADTC (datetime), FASTRESN (grams)]   reported meals
    df_bolus : DataFrame[FADTC (datetime), FASTRESN (units)]   bolus doses
    df_cgm   : DataFrame[datetime, cgm]  (optional; needed for bolus-only meals)
    train_cutoff : datetime; ICR is estimated from pairs BEFORE it (no leakage)
    size_source : 'reported' | 'bolus_icr' | 'hybrid' (default)
    Returns (df_meals[FADTC, FASTRESN, source], stats dict).
    """
    stats = dict(n_reported=0, n_after_merge=0, n_paired=0, n_reported_only=0,
                 n_bolus_only_added=0, n_bolus_only_dropped=0,
                 n_size_overridden=0, n_capped=0, icr=np.nan, icr_source="")

    has_c = df_carbs is not None and len(df_carbs) > 0
    has_b = df_bolus is not None and len(df_bolus) > 0
    meals = (df_carbs[["FADTC", "FASTRESN"]].dropna().copy() if has_c
             else _empty()[["FADTC", "FASTRESN"]])
    bolus = (df_bolus[["FADTC", "FASTRESN"]].dropna().copy() if has_b
             else _empty()[["FADTC", "FASTRESN"]])
    stats["n_reported"] = len(meals)

    meals = merge_close_records(meals)
    meals = meals[meals["FASTRESN"] > 0].reset_index(drop=True)
    bolus = bolus[bolus["FASTRESN"] > 0].sort_values("FADTC").reset_index(drop=True)
    stats["n_after_merge"] = len(meals)

    pairs = {}
    if len(meals) and len(bolus):
        pairs = _pair_one_to_one(meals["FADTC"].values.astype("datetime64[ns]"),
                                 bolus["FADTC"].values.astype("datetime64[ns]"),
                                 window_min)

    # --- per-subject insulin-to-carb ratio from matched pairs (train only) ---
    ratios = []
    for i, j in pairs.items():
        u = float(bolus.at[j, "FASTRESN"])
        if u >= MIN_BOLUS_U and (train_cutoff is None or meals.at[i, "FADTC"] < train_cutoff):
            ratios.append(float(meals.at[i, "FASTRESN"]) / u)
    if len(ratios) >= MIN_PAIRS_FOR_ICR:
        icr = float(np.clip(np.median(ratios), *ICR_RANGE))
        stats["icr_source"] = f"median of {len(ratios)} meal-bolus pairs"
    else:
        icr = DEFAULT_ICR
        stats["icr_source"] = f"default (only {len(ratios)} pairs)"
    stats["icr"] = icr

    rows = []
    paired_bolus = set(pairs.values())

    # 1) reported meals (paired with a bolus or not)
    for i in range(len(meals)):
        carbs = float(meals.at[i, "FASTRESN"])
        if i in pairs:
            j = pairs[i]
            u = float(bolus.at[j, "FASTRESN"])
            t = bolus.at[j, "FADTC"]                     # bolus timestamp
            est = u * icr
            if u >= MIN_BOLUS_U and size_source != "reported":
                if size_source == "bolus_icr":
                    carbs, stats["n_size_overridden"] = est, stats["n_size_overridden"] + 1
                elif carbs > HYBRID_RATIO * est or carbs < est / HYBRID_RATIO:
                    carbs, stats["n_size_overridden"] = est, stats["n_size_overridden"] + 1
            rows.append((t, carbs, "paired"))
            stats["n_paired"] += 1
        else:
            rows.append((meals.at[i, "FADTC"], carbs, "reported_only"))
            stats["n_reported_only"] += 1

    # 2) boluses with no reported meal -> "missed meal" if CGM rises afterwards
    cgm_t = cgm_v = None
    if df_cgm is not None and len(df_cgm):
        c = df_cgm.sort_values("datetime")
        cgm_t = c["datetime"].values.astype("datetime64[ns]")
        cgm_v = c["cgm"].values.astype(float)
    for j in range(len(bolus)):
        if j in paired_bolus:
            continue
        u = float(bolus.at[j, "FASTRESN"])
        t = bolus.at[j, "FADTC"]
        ok = u >= MIN_BOLUS_U and cgm_t is not None
        if ok:
            rise = _cgm_rise(cgm_t, cgm_v, t)
            ok = np.isfinite(rise) and rise >= MIN_RISE_MGDL
        if ok:
            rows.append((t, u * icr, "bolus_only"))
            stats["n_bolus_only_added"] += 1
        else:
            stats["n_bolus_only_dropped"] += 1

    out = (pd.DataFrame(rows, columns=["FADTC", "FASTRESN", "source"])
           if rows else _empty())
    if len(out):
        capped = out["FASTRESN"] > MAX_MEAL_G
        stats["n_capped"] = int(capped.sum())
        out["FASTRESN"] = out["FASTRESN"].clip(MIN_MEAL_G, MAX_MEAL_G)
        out = out.sort_values("FADTC").reset_index(drop=True)
    return out, stats


def print_stats(s):
    print(f"      Meal reconciliation: {s['n_reported']} reported -> "
          f"{s['n_after_merge']} after merging records <= {MERGE_GAP_MIN} min | "
          f"paired={s['n_paired']}, reported-only={s['n_reported_only']}, "
          f"bolus-only added={s['n_bolus_only_added']} "
          f"(dropped {s['n_bolus_only_dropped']}) | "
          f"size overridden={s['n_size_overridden']}, capped={s['n_capped']} | "
          f"ICR={s['icr']:.1f} g/U ({s['icr_source']})")