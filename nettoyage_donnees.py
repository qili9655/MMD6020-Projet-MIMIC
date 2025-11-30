import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import reduce
import re

# ============================================
#           Load relevant csvs
# ============================================
# ---------------------------------------------------------
# GLOBAL CACHE DICTIONARY
# ---------------------------------------------------------
_MIMIC_CACHE = {}

import duckdb

con = duckdb.connect()   # global DuckDB connection
_MIMIC_CACHE = {}         # keep your existing cache


def load_mimic_table(
    table_name,
    dataset="demo",
    section="hosp",
    base_path="/Users/qili/Library/CloudStorage/OneDrive-SanteetServicessociaux/Soins intensifs/Érudition/Recherche/MSc MedComp/MMD6020 Fondements médecine computationnelle/Projet",
    columns="*",
    where=None
):
    """
    Load a MIMIC-IV table using DuckDB (streaming, memory-safe), 
    optionally selecting columns or filtering rows.

    Returns a pandas DataFrame (small!) so all your other code works.
    """

    dataset_folders = {
        "demo": "mimic-iv-clinical-database-demo-2.2",
        "full": "mimic-iv-3.1"
    }

    if dataset not in dataset_folders:
        raise ValueError("dataset must be 'demo' or 'full'")
    if section not in ["hosp", "icu"]:
        raise ValueError("section must be 'hosp' or 'icu'")

    # --- Cache Key ---
    cache_key = (dataset, section, table_name, str(columns), str(where))
    if cache_key in _MIMIC_CACHE:
        print(f"→ Loaded from cache: {dataset}/{section}/{table_name}")
        return _MIMIC_CACHE[cache_key]

    folder = dataset_folders[dataset]
    file_path = os.path.join(base_path, folder, section, f"{table_name}.csv")

    print(f"→ Loading via DuckDB: {file_path}")

    # --- Build SQL query (streamed, no memory blow-up) ---
    sql = f"""
        SELECT {columns}
        FROM read_csv_auto('{file_path}')
    """

    if where is not None:
        sql += f" WHERE {where}"

    df = con.execute(sql).df()   # Convert only *result* to pandas

    print(f"→ Loaded {len(df):,} rows.")
    _MIMIC_CACHE[cache_key] = df
    return df

# ============================================
#           Define ITEMIDs needed
# ============================================

ITEMIDS = {
    # --------- VITAL SIGNS ---------
    'HR': [220045, 211],
    'RR': [220210, 618],
    'MAP': [220052, 456],
    'SPO2': [220277],
    'FIO2': [223835, 3420],
    'TEMP': [223761, 223762, 678],

    # --------- GCS ---------
    'GCS_E': [223900],
    'GCS_V': [223901],
    'GCS_M': [223902],

     # --------- LABS ---------
    "LABS": {
        "LACTATE": {
            "lab":  [50813],
            "chart": [225668]
        },

        "CREATININE": {
            "lab":  [50912],
            "chart": [229761]
        },

        "BUN": {
            "lab":  [51006],
            "chart": [225624]
        },

        "BICARB": {
            "lab":  [50882],
            "chart": [2121, 220645, 227443]
        },

        "SODIUM": {
            "lab":  [50824, 52455, 50983, 52623],
            "chart": [200645]          
        },

        "POTASSIUM": {
            "lab":  [52610, 50822, 50971, 52452],
            "chart": [227464]    
        },

        "MAGNESIUM": {
            "lab":  [50960],
            "chart": [220635]
        },

        "CALCIUM": {
            "lab":  [50893],
            "chart": [225625, 225667]     # non ionized and ionized calcium
        },

        "WBC": {
            "lab":  [51755, 51756, 51301],
            "chart": [220546]
        },

        "HEMOGLOBIN": {
            "lab":  [51222],
            "chart": [220228]     
        },

        "PLATELETS": {
            "lab":  [51265],
            "chart": [227457]
        },

        "BILIRUBIN": {
            "lab": [50885, 50883],
            "chart": [225690]
        },

        "INR": {
            "lab":  [51237],
            "chart": [227467, 220561]
        },

        "PAO2": {
            "lab":  [50821],
            "chart": [220224]
        }
    },
    "VENT_DURATION" : {
    # Ventilator mode indicators (patient is ON ventilator)
    "modes": [
        720,        # Ventilator Mode (old)
        223849,     # Ventilator Mode (new)
        223848,     # Vent Type
        223834      # MechVent flag
    ],

    # Ventilator settings (also indicate active ventilation)
    "settings": [
        224696,     # Tidal Volume (observed)
        224695,     # Tidal Volume (set)
        224697,     # Respiratory Rate (set)
        224746,     # PEEP
        224700,     # Plateau Pressure
        224701,     # Peak Inspiratory Pressure
        224702,     # Mean Airway Pressure
        224684,     # Pressure Support
        224685      # Pressure Control
    ],

    # Start of mechanical ventilation (INTUBATION)
    "intubation_events": [
        224385      # Intubation
    ],

    # End of mechanical ventilation (EXTUBATION)
    "extubation_events": [
        227194,     # Extubation
        225468,     # Unplanned Extubation (patient)
        225477      # Unplanned Extubation (non-patient)
    ],

    # Non-invasive ventilation (not an extubation)
    "noninvasive_events": [
        225794       # NIV / BiPAP / CPAP is *ventilation ON*, NOT OFF
    ]
    },

    # --------- VASOPRESSORS ---------
    'PRESSORS': {
        'norepi': 221906,
        'epi': 221289,
        'vaso': 221662,
        'dopamine': 222315,
        'dobutamine': [221653],
        'milrinone':  [221986, 221987],
        "phenylephrine": [229630, 229631, 221749, 229632, 229789],
    },
    # --------- ATB ---------
    'ANTIBIOTICS': {
        'vancomycin': 225798,
        'amikacin': 225840,
        'ampicillin': 225842,
        'ampicillin_sulbactam': 225843,   # Unasyn
        'azithromycin': 225845,
        'caspofungin': 225848,
        'cefazolin': 225850,
        'cefepime': 225851,
        'ceftazidime': 225853,
        'ceftriaxone': 225855,
        'clindamycin': 225860,
        'daptomycin': 225863,
        'fluconazole': 225869,
        'gentamicin': 225875,
        'imipenem_cilastatin': 225876,
        'linezolid': 225881,
        'meropenem': 225883,
        'metronidazole': 225884,
        'micafungin': 225885,
        'penicillin_g': 225890,
        'piperacillin': 225892,
        'piperacillin_tazobactam': 225893,  # Zosyn
        'tobramycin': 225902,
    },
    'SUPPORT': {
        'CRRT': [225128, 225441, 225955, 225805, 225803, 225802, 225436, 225809]
    }
}

# ======================================================
# Filter a time window by stay_id + charttime
# ======================================================
def filter_time_window(df, stay, hours):
    """
    Filters rows in df that fall within the last `hours` before ICU discharge.
    Uses stay_id if available, otherwise falls back to hadm_id.
    """

    t_end = stay.outtime
    t_start = t_end - pd.Timedelta(hours=hours)

    # --- Select correct key ---
    if 'stay_id' in df.columns:
        key = 'stay_id'
        value = stay.stay_id

    elif 'hadm_id' in df.columns:
        key = 'hadm_id'
        value = stay.hadm_id

    else:
        raise KeyError(
            "Neither 'stay_id' nor 'hadm_id' found in the dataframe. "
            "Cannot match rows to the ICU stay."
        )

    # --- Return filtered window ---
    return df[
        (df[key] == value) &
        (df.charttime >= t_start) &
        (df.charttime <= t_end)
    ]


# ==================================
# Safely converts to numeric
# ==================================

def to_numeric_safe(x):
    """
    Convert a charted value into a numeric value by extracting any digits
    from the string. Preserves real numeric values. Returns NaN only if 
    NO number is present.
    """

    if pd.isna(x):
        return np.nan

    # Already numeric
    if isinstance(x, (int, float, np.number)):
        return float(x)

    # Convert to string
    s = str(x).strip()

    # Extract the first number (integer or decimal)
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        return float(match.group())

    # No numeric content at all → return NaN
    return np.nan

# ==================================
# Last recorded value of a variable
# ==================================
def last_value(df):
    """Return last non-null valuenum after converting to numeric."""
    if df.empty:
        return np.nan

    # convert valuenum/value safely
    vals = pd.to_numeric(df["valuenum"], errors="coerce")
    if vals.notnull().any():
        return vals.iloc[-1]
    return np.nan

# ==================================
#       Get last GCS value
# ==================================
GCS_MOTOR_MAP = {
    # True motor response scale
    "Obeys Commands": 6,
    "Follows Commands": 6,
    "Normal": 6,

    "Localizes Pain": 5,

    "Withdraws from Pain": 4,
    "Withdraws": 4,
    "Withdraws to Pain": 4,

    "Flexion": 3,
    "Abnormal Flexion": 3,

    "Extension": 2,
    "Abnormal Extension": 2,

    "No response": 1,
    "None": 1,
    "No Response-ETT": 1,
    "Intubated/trached": 1,
    "Intubated": 1,
    "Intubated/Trached": 1,

    # Values we must treat as missing
    "Garbled": None,   # speech descriptor
    "Slurred": None,   # speech descriptor
}

def last_gcs_value(df, map_dict):
    """Return numeric GCS score from valuenum or mapped text."""
    if df.empty:
        return np.nan
    
    last = df.sort_values('charttime').iloc[-1]

    # If numeric is available → use it
    if pd.notnull(last['valuenum']):
        return float(last['valuenum'])

    # Otherwise use text mapping
    label = last['value']
    return map_dict.get(label, np.nan)

# ================================================
#       Building vitals at discharge
# ================================================
def build_vitals_discharge(icustays, chartevents, labevents, itemids, window_hours=24):
    """
    Extract last available vitals/labs in the final `window_hours` before ICU discharge.
    """
    rows = []

    for _, stay in icustays.iterrows():

        # Filter datasets
        ce   = filter_time_window(chartevents, stay, window_hours)
        labs = filter_time_window(labevents, stay, window_hours)

        row = {"stay_id": stay.stay_id}

        # ----------------------
        #       VITALS
        # ----------------------
        row["hr"]   = last_value(ce[ce.itemid.isin(itemids["HR"])])
        row["rr"]   = last_value(ce[ce.itemid.isin(itemids["RR"])])
        row["map"]  = last_value(ce[ce.itemid.isin(itemids["MAP"])])
        row["spo2"] = last_value(ce[ce.itemid.isin(itemids["SPO2"])])

        # FiO2 handling
        raw_fio2 = last_value(ce[ce.itemid.isin(itemids["FIO2"])])
        if pd.notnull(raw_fio2):
            row["fio2"] = raw_fio2/100 if raw_fio2 > 1 else raw_fio2
        else:
            row["fio2"] = np.nan

        # Temperature
        row["temp"] = last_value(ce[ce.itemid.isin(itemids["TEMP"])])

        # ----------------------
        #       GCS
        # ----------------------
        e = last_value(ce[ce.itemid.isin(itemids["GCS_E"])])
        v = last_value(ce[ce.itemid.isin(itemids["GCS_V"])])

        m = last_gcs_value(ce[ce.itemid.isin(itemids["GCS_M"])], GCS_MOTOR_MAP)

        if pd.notnull(e) and pd.notnull(v) and pd.notnull(m):
            row["gcs"] = e + v + m
        else:
            row["gcs"] = np.nan

        # ----------------------
        #       ABGs
        # ----------------------
        # Prefer LABEVENTS, fallback to CHARTEVENTS
        lab_pao2_ids   = itemids["LABS"]["PAO2"]["lab"]
        chart_pao2_ids = itemids["LABS"]["PAO2"]["chart"]

        pao2_lab   = last_value(labs[labs.itemid.isin(lab_pao2_ids)])
        pao2_chart = last_value(ce[ce.itemid.isin(chart_pao2_ids)])

        # Prefer lab if available
        pao2 = pao2_lab if pd.notnull(pao2_lab) else pao2_chart
        row["pao2"] = pao2


        fio2 = row["fio2"]
        if pd.notnull(pao2) and pd.notnull(fio2) and fio2 > 0:
            row["pfratio"] = pao2 / fio2
        else:
            row["pfratio"] = np.nan

        rows.append(row)
        
    return pd.DataFrame(rows)


# ====================================
#       Ventilation duration
# ====================================

def compute_ventilation_duration(chartevents, icustays, vent_ids):
    """
    Computes mechanical ventilation duration using explicit intubation/extubation
    events plus ventilator mode/settings fallback.

    Adds:
        - vent_hours
        - vent_days  (vent_hours / 24)
    """

    # Extract the relevant IDs
    modes      = set(vent_ids["modes"])
    settings   = set(vent_ids["settings"])
    intub_ids  = set(vent_ids["intubation_events"])
    extub_ids  = set(vent_ids["extubation_events"])
    niv_ids    = set(vent_ids["noninvasive_events"])

    # Keep only rows that might matter
    useful_ids = modes | settings | intub_ids | extub_ids | niv_ids
    df = chartevents[chartevents["itemid"].isin(useful_ids)].copy()

    df = df.sort_values(["stay_id", "charttime"])

    results = []

    for stay_id, group in df.groupby("stay_id"):
        group = group.sort_values("charttime")

        # ICU discharge time
        t_out = icustays.loc[icustays["stay_id"] == stay_id, "outtime"]
        if t_out.empty:
            continue
        t_out = t_out.iloc[0]

        # Track ventilation state
        vent_on = False
        t_start = None
        total_hours = 0.0

        for _, row in group.iterrows():
            item = row.itemid
            t = row.charttime

            # ---- INTUBATION EVENT → ventilation ON ----
            if item in intub_ids:
                if not vent_on:
                    vent_on = True
                    t_start = t

            # ---- EXTUBATION EVENT → ventilation OFF ----
            elif item in extub_ids:
                if vent_on:
                    total_hours += (t - t_start).total_seconds() / 3600
                    vent_on = False
                    t_start = None

            # ---- Ventilator MODE/SETTINGS → implicit ON ----
            elif item in modes or item in settings:
                if not vent_on:
                    vent_on = True
                    t_start = t

            # ---- NIV events → ignore ----
            else:
                pass

        # If still ventilated at ICU discharge → close interval
        if vent_on and t_start is not None:
            total_hours += (t_out - t_start).total_seconds() / 3600

        results.append({
            "stay_id": stay_id,
            "vent_hours": total_hours
        })

    # Convert to DataFrame
    df_result = pd.DataFrame(results)

    # Add vent_days
    df_result["vent_days"] = df_result["vent_hours"] / 24

    return df_result

# ====================================
#       Duration of ICU stay 
# ====================================
def compute_icu_los(icustays):
    """
    Adds ICU length of stay to icustays dataframe.
    Returns a copy with columns:
        - los_hours
        - los_days
    """
    df = icustays.copy()

    df["los_hours"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    df["los_days"] = df["los_hours"] / 24

    return df

# ====================================
#       Recent organ support 
# ====================================

def build_supports_recent(
    icustays,
    chartevents,
    inputevents,
    procedureevents,
    ITEMIDS,
    window_hours=72,
    extubation_ids=None,
    crrt_ids=None
):
    """
    Compute support status (ventilation, vasopressors, extubation, CRRT)
    in the last X hours before ICU discharge.
    """

    if extubation_ids is None:
        extubation_ids = ITEMIDS["VENT_DURATION"]["extubation_events"]

    if crrt_ids is None:
        crrt_ids = ITEMIDS["SUPPORT"]["CRRT"]

    rows = []

    for _, stay in icustays.iterrows():

        # Time window
        ce_window = filter_time_window(chartevents, stay, window_hours)

        iv_window = inputevents[
            (inputevents.stay_id == stay.stay_id) &
            (inputevents.starttime <= stay.outtime) &
            (inputevents.endtime >= stay.outtime - pd.Timedelta(hours=window_hours))
        ]

        proc_window = procedureevents[
            (procedureevents.stay_id == stay.stay_id) &
            (procedureevents.endtime.between(
                stay.outtime - pd.Timedelta(hours=window_hours),
                stay.outtime
            ))
        ]

        row = {"stay_id": stay.stay_id}

        # --- Mechanical ventilation ON ---
        row["vent_last48"] = int(
            ce_window.itemid.isin(ITEMIDS["VENT_DURATION"]["modes"]).any()
            or ce_window.itemid.isin(ITEMIDS["VENT_DURATION"]["settings"]).any()
            or proc_window.itemid.isin(ITEMIDS["VENT_DURATION"]["intubation_events"]).any()
        )

        # --- Vasopressors ---
        pressor_ids = list(ITEMIDS["PRESSORS"].values())
        row["pressor_last48"] = int(iv_window.itemid.isin(pressor_ids).any())

        # --- Extubation events ---
        row["extubation_last48"] = int(proc_window.itemid.isin(extubation_ids).any())

        # --- CRRT ---
        row["crrt_last48"] = int(proc_window.itemid.isin(crrt_ids).any())

        rows.append(row)

    return pd.DataFrame(rows)


# ====================================
#       labs at discharge 
# ====================================
def build_labs_from_labevents(icustays, labevents, ITEMIDS, window_hours=72):
    """
    Extract last lab values from LABEVENTS only.
    """
    rows = []
    LABS = ITEMIDS["LABS"]

    for _, stay in icustays.iterrows():
        w = filter_time_window(labevents, stay, window_hours)
        row = {"stay_id": stay.stay_id}

        for labname, sources in LABS.items():
            lab_ids = sources["lab"]
            row[labname.lower() + "_lab"] = last_value(w[w.itemid.isin(lab_ids)])

        rows.append(row)

    return pd.DataFrame(rows)


def build_labs_from_chartevents(icustays, chartevents, ITEMIDS, window_hours=72):
    """
    Extract last lab-like values from CHARTEVENTS only.
    (ABGs, point-of-care tests, ionized Ca, ABG Hb, etc.)
    """
    rows = []
    LABS = ITEMIDS["LABS"]

    for _, stay in icustays.iterrows():
        w = filter_time_window(chartevents, stay, window_hours)
        row = {"stay_id": stay.stay_id}

        for labname, sources in LABS.items():
            chart_ids = sources["chart"]
            row[labname.lower() + "_chart"] = last_value(w[w.itemid.isin(chart_ids)])

        rows.append(row)

    return pd.DataFrame(rows)


def fuse_final_labs(df_labs_discharge, ITEMIDS):
    """
    Produces a clean dataframe with ONE final lab column per test.

    - Uses ONLY lab values from LABEVENTS
    - Drops *_lab and *_chart columns
    - Keeps only: stay_id + final labname columns
    """

    LABS = ITEMIDS["LABS"]
    df = df_labs_discharge.copy()

    final_cols = ["stay_id"]

    for labname in LABS.keys():
        lab_col   = labname.lower() + "_lab"
        final_col = labname.lower()

        # Final lab value = labevents only
        df[final_col] = df[lab_col]

        final_cols.append(final_col)

    # return only: stay_id + final lab columns
    return df[final_cols]

# ====================================
#       Compute Delta 
# ====================================

def compute_delta_48h(df, ids):
    """
    Compute:
    - last_value: most recent measurement in entire ICU stay
    - prev_value: measurement at least 48h before last_value
    - delta = last_value - prev_value
    """

    sub = df[df.itemid.isin(ids)].copy()
    if sub.empty:
        return np.nan

    # Ensure numeric
    sub["valuenum"] = pd.to_numeric(sub["valuenum"], errors="coerce")

    # Drop missing or non-numeric
    sub = sub.dropna(subset=["valuenum"])
    if sub.empty:
        return np.nan

    # 1) Find the LAST measurement
    sub = sub.sort_values("charttime")
    last_row = sub.iloc[-1]
    last_time = last_row["charttime"]
    last_value = last_row["valuenum"]

    # 2) Find the most recent value at least 48h before last_time
    cutoff = last_time - pd.Timedelta(hours=48)
    prev_candidates = sub[sub["charttime"] <= cutoff]

    if prev_candidates.empty:
        return np.nan

    prev_value = prev_candidates.iloc[-1]["valuenum"]

    return last_value - prev_value

# ====================================
#       Delta vitals & labs 
# ====================================
def build_trajectories_48h(
    icustays,
    labevents,
    chartevents,
    ITEMIDS
):
    LABS = ITEMIDS["LABS"]
    VITAL_KEYS = ["HR", "RR", "MAP", "SPO2", "FIO2", "TEMP"]

    rows = []

    for _, stay in icustays.iterrows():

        # ------------------------------------------
        # LABEVENTS for this ICU stay
        # Match *by hospitalization + time window*
        # ------------------------------------------
        df_lab = labevents[
            (labevents.subject_id == stay.subject_id) &
            (labevents.hadm_id == stay.hadm_id) &
            (labevents.charttime >= stay.intime - pd.Timedelta(hours=48)) &
            (labevents.charttime <= stay.outtime)
        ].copy()

        # ------------------------------------------
        # CHARTEVENTS for this ICU stay (vitals)
        # ------------------------------------------
        df_char = chartevents[
            (chartevents.stay_id == stay.stay_id) &
            (chartevents.charttime >= stay.intime - pd.Timedelta(hours=48)) &
            (chartevents.charttime <= stay.outtime)
        ].copy()

        row = {"stay_id": stay.stay_id}

        # ======================================================
        #                   LAB DELTAS (LABEVENTS ONLY)
        # ======================================================
        for labname, ids_dict in LABS.items():
            lab_ids = ids_dict["lab"]     # ✔ only lab itemids

            delta = compute_delta_48h(df_lab, lab_ids)
            row[f"delta_{labname.lower()}"] = delta

        # ======================================================
        #                  VITAL DELTAS (CHARTEVENTS)
        # ======================================================
        for vital in VITAL_KEYS:
            vital_ids = ITEMIDS[vital]

            delta = compute_delta_48h(df_char, vital_ids)
            row[f"delta_{vital.lower()}"] = delta

        rows.append(row)

    return pd.DataFrame(rows)


# ====================================
#       Medications 
# ====================================
def build_meds_recent(
    icustays,
    inputevents,
    prescriptions,
    ITEMIDS,
    window_hours=72
):
    """
    Extract recent medication exposure (last window_hours before ICU discharge).
    Includes:
      - Vasopressors (from inputevents using ITEMIDS['PRESSORS'])
      - Antibiotics (from prescriptions using ITEMIDS['ANTIBIOTICS'])
      - Sedatives 
      - Steroids (hydrocortisone, methylpred)
      - Diuretics (furosemide, bumetanide, acetazolamide)

    Returns one row per stay_id.
    """

    # Build regex patterns ------------------------------------------

    # Antibiotics → use ALL drug names from dictionary
    antibiotic_keywords = "|".join(ITEMIDS["ANTIBIOTICS"].keys())

    # Sedatives (expanded)
    sedative_patterns = [
        "midazolam",
        "lorazepam",
        "propofol",
        "dexmed", "dexme",  # dexmedetomidine
        "ketamine"
    ]
    sedative_regex = "|".join(sedative_patterns)

    # Steroids
    steroid_patterns = [
        "hydrocortisone",
        "methylpred", "methylprednisolone"
    ]
    steroid_regex = "|".join(steroid_patterns)

    # Diuretics
    diuretic_patterns = [
        "furosemide",
        "bumetanide",
        "acetazolamide"
    ]
    diuretic_regex = "|".join(diuretic_patterns)

    # Vasopressor ItemIDs
    pressor_itemids = []
    for v in ITEMIDS["PRESSORS"].values():
        if isinstance(v, list):
            pressor_itemids.extend(v)
        else:
            pressor_itemids.append(v)

    rows = []

    # =================== LOOP OVER ICU STAYS =================================
    for _, stay in icustays.iterrows():

        # ------------ IV medications (inputevents) ---------------------
        iv_window = inputevents[
            (inputevents.stay_id == stay.stay_id) &
            (inputevents.starttime <= stay.outtime) &
            (inputevents.endtime >= stay.outtime - pd.Timedelta(hours=window_hours))
        ]

        # ------------ Prescription medications (hospital RX) -----------
        rx_window = prescriptions[
            (prescriptions.hadm_id == stay.hadm_id) &
            (prescriptions.starttime >= stay.outtime - pd.Timedelta(hours=window_hours)) &
            (prescriptions.starttime <= stay.outtime)
        ].copy()

        row = {"stay_id": stay.stay_id}

        # ======================== MEDICATION FLAGS ===============================

        # -- Vasopressors (IV only)
        row["pressor_last72"] = int(iv_window.itemid.isin(pressor_itemids).any())

        # -- Antibiotics (RX only, using your dictionary keys)
        row["antibiotics_last72"] = int(
            rx_window["drug"].str.contains(antibiotic_keywords, case=False, na=False).any()
        )

        # -- Sedatives (RX)
        row["sedatives_last72"] = int(
            rx_window["drug"].str.contains(sedative_regex, case=False, na=False).any()
        )

        # -- Steroids
        row["steroids_last72"] = int(
            rx_window["drug"].str.contains(steroid_regex, case=False, na=False).any()
        )

        # -- Diuretics
        row["diuretics_last72"] = int(
            rx_window["drug"].str.contains(diuretic_regex, case=False, na=False).any()
        )

        rows.append(row)

    return pd.DataFrame(rows)

# ====================================
#       Final dataframe
# ====================================

def build_final_dataset(
    df_demographics,
    df_vitals_discharge,
    df_supports_recent,
    df_labs_discharge,
    df_trajectories_48h,
    df_meds_recent,
    df_operational,
    vent_los_df=None,         
):
    """
    Merge all ICU-level dataframes using stay_id,
    apply exclusions (vent <48h, ICU LOS>30 days).
    """

    # ---- 1. Collect all incoming dfs ----
    dfs = [
        df_demographics,
        df_vitals_discharge,
        df_supports_recent,
        df_labs_discharge,
        df_trajectories_48h,
        df_meds_recent,
        df_operational
    ]

    # ---- 2. Drop None dfs to avoid merge errors ----
    dfs = [df for df in dfs if df is not None]

    # ---- 3. Merge everything on stay_id ----
    df_final = reduce(
        lambda left, right: left.merge(right, on="stay_id", how="left"),
        dfs
    )

    # ---- 4. Optionally merge LOS + ventilation durations ----
    if vent_los_df is not None:
        df_final = df_final.merge(vent_los_df, on="stay_id", how="left")

    # ---- 5. Exclusion criteria ----
    # Ventilation <48h
    if "vent_hours" in df_final.columns:
        df_final = df_final[df_final["vent_hours"] >= 48]

    # ICU LOS > 30 days  (30 * 24 = 720 hours)
    if "los_hours" in df_final.columns:
        df_final = df_final[df_final["los_hours"] <= 720]

    # Final tidy index
    df_final = df_final.reset_index(drop=True)

    return df_final



# ====================================
#       Charlson
# ====================================

CHARLSON_GROUPS_ICD9 = {
    "myocardial_infarction": ["410", "412"],
    "congestive_heart_failure": ["398", "402", "404", "428"],
    "peripheral_vascular_disease": ["440", "441", "443", "447", "557", "V43"],
    "cerebrovascular_disease": ["430", "431", "432", "433", "434", "435", "436", "437", "438"],
    "dementia": ["290"],
    "chronic_pulmonary_disease": ["490", "491", "492", "493", "494", "495", "496"],
    "rheumatic_disease": ["446", "701", "710", "711", "714", "719"],
    "peptic_ulcer_disease": ["531", "532", "533", "534"],
    "mild_liver_disease": ["570", "571"],
    "diabetes_without_complications": ["250"],
    "diabetes_with_complications": ["250"],  # differentiated later by 4th/5th digit, optional
    "hemiplegia_paraplegia": ["342", "343", "344"],
    "renal_disease": ["582", "583", "585", "586", "588"],
    "malignancy": ["140", "141", "142", "143", "144", "145", "146", "147", "148", "149",
                   "150","151","152","153","154","155","156","157","158","159","160","161",
                   "162","163","164","165","170","171","172","174","175","176"],
    "severe_liver_disease": ["572"],
    "metastatic_solid_tumor": ["196", "197", "198", "199"],
    "aids_hiv": ["042", "043", "044"]
}

CHARLSON_GROUPS_ICD10 = {
    "myocardial_infarction": ["I21","I22","I252"],
    "congestive_heart_failure": ["I50","I110","I130","I132"],
    "peripheral_vascular_disease": ["I70","I71","I731","I738","I739","I771","I790","I792","K551","K558","K559","Z958","Z959"],
    "cerebrovascular_disease": ["I60","I61","I62","I63","I64","I65","I66","I67","I68","I69","G45","G46"],
    "dementia": ["F00","F01","F02","F03","G30"],
    "chronic_pulmonary_disease": ["J40","J41","J42","J43","J44","J45","J46","J47","J60","J61","J62","J63","J64","J65","J66","J67"],
    "rheumatic_disease": ["M05","M06","M315","M32","M33","M34","M351","M353","M360"],
    "peptic_ulcer_disease": ["K25","K26","K27","K28"],
    "mild_liver_disease": ["B18","K700","K701","K702","K703","K709","K713","K714","K715","K717"],
    "diabetes_without_complications": ["E100","E101","E109","E110","E111","E119","E120","E121","E129","E130","E131","E139","E140","E141","E149"],
    "diabetes_with_complications": ["E102","E104","E105","E107","E112","E114","E115","E117","E122","E124","E125","E127","E132","E134","E135","E137","E142","E144","E145","E147"],
    "hemiplegia_paraplegia": ["G81","G82","G041"],
    "renal_disease": ["N18","N19","N052","N053","N054","N055","N056","N057","N250","Z490","Z491","Z492","Z940","Z992"],
    "malignancy": ["C00","C01","C02","C03","C04","C05","C06","C07","C08","C09","C10","C11","C12","C13","C14","C15",
                   "C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C30","C31","C32","C33","C34",
                   "C37","C38","C39","C40","C41","C43","C45","C46","C47","C48","C49","C50","C51","C52","C53","C54",
                   "C55","C56","C57","C58","C60","C61","C62","C63","C64","C65","C66","C67","C68","C69","C70","C71",
                   "C72","C73","C74","C75","C76","C81","C82","C83","C84","C85","C88","C90","C91","C92","C93","C94","C95","C96"],
    "severe_liver_disease": ["I850","I859","I864","I982","K704","K711","K721","K729"],
    "metastatic_solid_tumor": ["C77","C78","C79","C80"],
    "aids_hiv": ["B20","B21","B22","B24"]
}

# Weight of each Charlson category
CHARLSON_WEIGHTS = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "chronic_pulmonary_disease": 1,
    "rheumatic_disease": 1,
    "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1,
    "diabetes_without_complications": 1,
    "diabetes_with_complications": 2,
    "hemiplegia_paraplegia": 2,
    "renal_disease": 2,
    "malignancy": 2,
    "severe_liver_disease": 3,
    "metastatic_solid_tumor": 6,
    "aids_hiv": 6
}

def compute_and_merge_charlson(df_final, diagnoses):
    """
    Computes Charlson Comorbidity Index (CCI) from diagnoses table
    and merges into df_final using hadm_id.

    Parameters
    ----------
    df_final : DataFrame
        Final merged dataset containing 'hadm_id' column.
    diagnoses : DataFrame
        MIMIC diagnoses_icd table (must contain: hadm_id, icd_code, icd_version)

    Returns
    -------
    DataFrame
        df_final with a new column 'charlson_index'
    """

    # --------------------------
    # Helper: convert ICD codes
    # --------------------------
    def extract_icd_prefix(code, version):
        if pd.isnull(code):
            return None
        code = str(code)

        # ICD-10: first 3 chars (letter + 2 digits)
        if version == 10:
            return code[:3].upper()

        # ICD-9: before decimal, take first 3 digits
        if version == 9:
            return code.split('.')[0][:3]

        return None

    # Add prefix column
    diagnoses = diagnoses.copy()
    diagnoses["icd_prefix"] = diagnoses.apply(
        lambda r: extract_icd_prefix(r.icd_code, r.icd_version),
        axis=1
    )

    # --------------------------
    # Compute Charlson per HADM
    # --------------------------
    def compute_charlson_for_hadm(df_diag):
        score = 0

        # ICD-10
        prefixes10 = df_diag[df_diag.icd_version == 10]["icd_prefix"].unique().tolist()
        for cat, code_list in CHARLSON_GROUPS_ICD10.items():
            if any(prefix in code_list for prefix in prefixes10):
                score += CHARLSON_WEIGHTS[cat]

        # ICD-9
        prefixes9 = df_diag[df_diag.icd_version == 9]["icd_prefix"].unique().tolist()
        for cat, code_list in CHARLSON_GROUPS_ICD9.items():
            if any(prefix in code_list for prefix in prefixes9):
                score += CHARLSON_WEIGHTS[cat]

        return score

    # Compute Charlson grouped by admission
    charlson_scores = (
        diagnoses
        .groupby("hadm_id")
        .apply(compute_charlson_for_hadm)
        .reset_index()
        .rename(columns={0: "charlson_index"})
    )

    # --------------------------
    # Merge into df_final
    # --------------------------
    df_final = df_final.merge(charlson_scores, on="hadm_id", how="left")

    # If no diagnoses → Charlson = 0
    df_final["charlson_index"] = df_final["charlson_index"].fillna(0).astype(int)

    return df_final



# ====================================
#       Charlson
# ====================================

def add_readmit_72h(df_final, icustays):
    """
    Computes ICU readmission within 72 hours for each stay_id and 
    returns df_final with a new binary column 'readmit_72h'.
    
    Definition:
    readmit_72h = 1 if the next ICU stay (same hadm_id) starts within
    72 hours after the current stay's outtime.
    """

    # --- 1. Sort ICU stays ---
    icu = icustays.sort_values(["subject_id", "hadm_id", "intime"]).copy()
    icu["readmit_72h"] = 0

    # --- 2. Compute outcome within each hospital admission ---
    for (subject, hadm), group in icu.groupby(["subject_id", "hadm_id"]):
        group = group.sort_values("intime").reset_index(drop=True)

        for i in range(len(group) - 1):
            current = group.loc[i]
            nextstay = group.loc[i + 1]

            delta = nextstay["intime"] - current["outtime"]

            if pd.Timedelta("0h") <= delta <= pd.Timedelta("72h"):
                icu.loc[icu.stay_id == current.stay_id, "readmit_72h"] = 1

    # --- 3. Keep only stay_id + readmission ---
    df_readmit = icu[["stay_id", "readmit_72h"]]

    # --- 4. Merge into df_final ---
    df_final = df_final.merge(df_readmit, on="stay_id", how="left")

    # Safety: missing values become 0 (no readmission)
    df_final["readmit_72h"] = df_final["readmit_72h"].fillna(0).astype(int)

    return df_final


# ==================================================================================
#       Dataframe to use in models, with only predictor and outcome variables
# ==================================================================================

def clean_df(df):
    """
    Reduce df_final to the predictor + outcome columns needed for modeling.
    Columns missing from df are silently ignored.
    """

    keep_cols = [
        "stay_id", "insurance", "race", "gender", "age", 
        "spo2", "fio2", "temp", "map", "gcs", "pao2", "pfratio",
        "vent_last48", "pressor_last48", "extubation_last48", "crrt_last48",
        "intime", "outime", "subject_id",

        # Labs
        "lactate_lab", "creatinine_lab", "bun_lab", "bicarb_lab",
        "sodium_lab", "potassium_lab", "magnesium_lab", "calcium_lab",
        "wbc_lab", "hemoglobin_lab", "platelets_lab",
        "bilirubin_lab", "inr_lab", "pao2_lab",

        # Deltas
        "delta_creat", "delta_bun", "delta_bicarb", "delta_sodium",
        "delta_potassium", "delta_magnesium", "delta_calcium", "delta_wbc",
        "delta_hemoglobin", "delta_platelets", "delta_bilirubin",
        "delta_inr", "delta_pao2",
        "delta_hr", "delta_rr", "delta_map", "delta_spo2",
        "delta_fio2", "delta_temp",

        # Medication exposures
        "pressor_last72", "antibiotics_last72", "sedatives_last72",
        "steroids_last72", "diuretics_last72",

        # Operational
        "night_discharge", "weekend_discharge",
        "first_careunit", "last_careunit",

        # LOS + ventilation
        "los_days", "vent_days",

        # Indices/outcomes
        "charlson_index", "readmit_72h"
    ]

    # Keep only columns that actually exist
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Return clean dataframe
    return df[keep_cols].copy()