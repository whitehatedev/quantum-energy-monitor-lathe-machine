"""
data_clean.py
Cleans dataset.csv (produced by data_collect.py) → dataset_clean.csv

Steps:
- Ensures all expected columns exist
- Converts vibration intensity labels ('NO VIBRATION'/'LOW'/...) → numeric
- Coerces all sensor columns to numeric
- Sorts by timestamp
- Drops rows where every sensor field is NaN
- Forward-fills, then backward-fills remaining gaps
- Saves dataset_clean.csv (the file train.py reads)
"""

import os
import numpy as np
import pandas as pd

RAW_FILE   = "dataset.csv"
CLEAN_FILE = "dataset_clean.csv"

# --------------------------------------------------------------
#  Column definitions (must match data_collect.py output)
# --------------------------------------------------------------
PHASE_COLS = []
for p in ("R", "S", "T"):
    PHASE_COLS += [
        f"{p}_voltage", f"{p}_current", f"{p}_power",
        f"{p}_energy",  f"{p}_frequency", f"{p}_pf",
    ]

SENSOR_COLS = [
    "temperature_c",
    "vib_digital_count",
    "vib_intensity",     # string label → converted to numeric
    "piezo_intensity",
] + PHASE_COLS

# Vibration label → numeric severity (aligns with train.py's 0–2 range)
VIB_LABEL_MAP = {
    "no vibration": 0.0,
    "no_vibration": 0.0,
    "none":         0.0,
    "low":          0.5,
    "medium":       1.0,
    "mid":          1.0,
    "high":         2.0,
}


# --------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------
def convert_vib_labels(series: pd.Series) -> pd.Series:
    """Convert string vibration labels → numeric severity."""
    def _convert(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        key = str(v).strip().lower()
        if key in VIB_LABEL_MAP:
            return VIB_LABEL_MAP[key]
        # Match prefixes like "high (12)" or "low - 3 hits"
        for label, num in VIB_LABEL_MAP.items():
            if key.startswith(label):
                return num
        return np.nan

    return series.apply(_convert)


# --------------------------------------------------------------
#  Main cleaner
# --------------------------------------------------------------
def clean():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"{RAW_FILE} not found — run data_collect.py first."
        )

    df = pd.read_csv(RAW_FILE)
    print(f"📂 Loaded {RAW_FILE}: shape={df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    if df.empty:
        print("❌ dataset.csv is empty — nothing to clean.")
        return

    # 1. Ensure every expected column exists
    for col in SENSOR_COLS:
        if col not in df.columns:
            df[col] = np.nan
            print(f"   + Added missing column: {col}")

    # 2. Sort by timestamp
    if "timestamp_unix" in df.columns:
        df["timestamp_unix"] = pd.to_numeric(df["timestamp_unix"], errors="coerce")
        df = df.sort_values("timestamp_unix").reset_index(drop=True)
    elif "timestamp" in df.columns:
        df["_ts_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("_ts_dt").reset_index(drop=True)
        df.drop(columns=["_ts_dt"], inplace=True)
    print(f"   ↕ Sorted chronologically")

    # 3. Coerce all numeric columns (strings → NaN → handled below)
    numeric_cols = [
        c for c in PHASE_COLS
        + ["temperature_c", "vib_digital_count", "piezo_intensity"]
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Convert vibration label → numeric
    if "vib_intensity" in df.columns:
        df["vib_intensity"] = convert_vib_labels(df["vib_intensity"])

    # 5. Drop rows where EVERY sensor is missing
    before = len(df)
    df = df.dropna(subset=SENSOR_COLS, how="all")
    dropped = before - len(df)
    if dropped:
        print(f"   - Dropped {dropped} all-empty rows")

    if df.empty:
        print("❌ No usable rows left. Check the ESP32 is uploading data.")
        return

    # 6. Fill gaps: forward-fill, then backward-fill for leading NaNs
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().bfill()

    # 7. Any column still entirely NaN → fill with 0
    for col in SENSOR_COLS:
        if df[col].isna().all():
            df[col] = 0.0
            print(f"   ! {col} was entirely NaN → filled with 0")

    # 8. Round long decimals (keeps file smaller & cleaner)
    for col in SENSOR_COLS:
        df[col] = df[col].astype(float).round(4)

    # 9. Final column order
    keep_order = [
        "timestamp", "timestamp_unix",
        "temperature_c", "vib_digital_count",
        "vib_intensity", "piezo_intensity",
    ] + PHASE_COLS
    keep_order = [c for c in keep_order if c in df.columns]
    df_clean = df[keep_order].copy()

    # 10. Drop any row still containing NaN in a sensor field
    df_clean = df_clean.dropna(subset=SENSOR_COLS, how="any")

    # 11. Save
    df_clean.to_csv(CLEAN_FILE, index=False)

    print(f"\n✅ Saved cleaned dataset → {CLEAN_FILE}")
    print(f"   Shape: {df_clean.shape}")
    print(f"   Columns: {df_clean.columns.tolist()}")

    print("\n🔍 Missing values per column (should all be 0):")
    print(df_clean[SENSOR_COLS].isna().sum().to_string())

    print("\n📊 First 3 rows:")
    print(df_clean.head(3).to_string())

    print("\n📊 Basic statistics:")
    print(df_clean[SENSOR_COLS].describe().T.round(2).to_string())


# --------------------------------------------------------------
if __name__ == "__main__":
    try:
        clean()
    except Exception as e:
        print(f"❌ Clean failed: {e}")
        raise