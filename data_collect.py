"""
data_collect.py
Fetches ALL data from the ESP32's Firebase root node and flattens it into a CSV.

Captures:
  • temperature (°C)
  • vibration (count + intensity + piezo)
  • Phase R/S/T — voltage, current, power, energy, frequency, PF

Requires: serviceAccountKey.json (Firebase Admin SDK)
"""

import os
import time
from datetime import datetime

import requests
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np

# ==============================================================
#  CONFIG
# ==============================================================
DATABASE_URL    = "https://new-project-210e3-default-rtdb.firebaseio.com"
API_KEY         = "AIzaSyC2A3qi7vfPHr83Kev2TGyJBiSGMM7kmYs"
USER_EMAIL      = "sb284160@gmail.com"
USER_PASSWORD   = "Password@1"

SERVICE_ACCOUNT = "serviceAccountKey.json"
ROOT_PATH       = "/"

COLLECT_INTERVAL     = 1.0     # seconds between polls
COLLECT_DURATION_MIN = 10      # minutes per run (set to None → run until Ctrl+C)
OUTPUT_CSV           = "dataset.csv"   # matches data_clean.py

SIGN_IN_USER = True            # optional user sign-in (needed if rules require auth)
VERBOSE      = True            # set False to reduce print spam


# ==============================================================
#  AUTHENTICATION
# ==============================================================
def init_firebase():
    if not os.path.exists(SERVICE_ACCOUNT):
        raise FileNotFoundError(
            f"{SERVICE_ACCOUNT} not found. "
            f"Download it from Firebase Console → Project settings → Service accounts."
        )

    cred = credentials.Certificate(SERVICE_ACCOUNT)
    if firebase_admin._apps:
        firebase_admin.get_app()
    else:
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})
    print(f"✅ Service account authenticated: {cred.project_id}")

    if SIGN_IN_USER:
        try:
            url = (f"https://identitytoolkit.googleapis.com/v1/"
                   f"accounts:signInWithPassword?key={API_KEY}")
            resp = requests.post(url, json={
                "email": USER_EMAIL,
                "password": USER_PASSWORD,
                "returnSecureToken": True,
            }, timeout=20)
            if resp.status_code == 200:
                print(f"✅ User signed in: {USER_EMAIL}")
            else:
                print(f"⚠️  User sign-in failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️  User sign-in error: {e}")

    try:
        probe = db.reference(ROOT_PATH).get()
        if probe is None:
            print("⚠️  Probe read returned None — root node empty. Is the ESP32 running?")
        else:
            print(f"✅ Probe read OK. Firebase root currently has: {list(probe.keys())}")
    except Exception as e:
        raise RuntimeError(
            f"❌ Cannot read database: {e}\n   Check credentials and rules."
        ) from e


# ==============================================================
#  FETCH & FLATTEN
# ==============================================================
def fetch_snapshot(max_retries=3, retry_delay=2):
    ref = db.reference(ROOT_PATH)
    for attempt in range(1, max_retries + 1):
        try:
            return ref.get()
        except Exception as e:
            print(f"⚠️  Fetch attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    return None


def flatten_snapshot(raw):
    """Flatten the ESP32's nested JSON into a single flat row dict."""
    if raw is None:
        return None

    row = {}
    now = datetime.now()
    row["timestamp_unix"] = time.time()
    row["timestamp"]      = now.strftime("%Y-%m-%d %H:%M:%S")

    # --- VIBRATION ---
    vib = raw.get("vibration") or {}
    row["vib_count"]           = vib.get("count", np.nan)
    row["vib_intensity"]       = vib.get("intensity", None)          # string label
    row["piezo_intensity"]     = vib.get("piezoIntensity", np.nan)   # numeric
    row["vibration_intensity"] = row["piezo_intensity"]              # alias (legacy)

    # --- TEMPERATURE ---
    temp = raw.get("temperature") or {}
    row["temperature"] = temp.get("celsius", np.nan)

    # --- THREE PHASES ---
    for phase in ("R", "S", "T"):
        p = raw.get(f"phase{phase}") or {}
        row[f"{phase}_voltage"]   = p.get("voltage",     np.nan)
        row[f"{phase}_current"]   = p.get("current",     np.nan)
        row[f"{phase}_power"]     = p.get("power",       np.nan)
        row[f"{phase}_energy"]    = p.get("energy",      np.nan)
        row[f"{phase}_frequency"] = p.get("frequency",   np.nan)
        row[f"{phase}_pf"]        = p.get("powerFactor", np.nan)

    # --- ALIASES expected by data_clean.py / train.py ---
    row["temperature_c"]      = row["temperature"]
    row["vib_digital_count"]  = row["vib_count"]

    return row


# ==============================================================
#  COLLECT LOOP
# ==============================================================
def collect_to_csv(interval_sec=COLLECT_INTERVAL,
                   duration_min=COLLECT_DURATION_MIN,
                   output=OUTPUT_CSV):
    records = []
    end_time = (time.time() + duration_min * 60) if duration_min else None
    if duration_min:
        total_expected = int(duration_min * 60 / interval_sec)
        print(f"\nCollecting for {duration_min} min at {interval_sec}s "
              f"intervals (~{total_expected} samples)...\n")
    else:
        print(f"\nCollecting indefinitely at {interval_sec}s intervals. "
              f"Press Ctrl+C to stop.\n")

    try:
        while end_time is None or time.time() < end_time:
            raw = fetch_snapshot()
            row = flatten_snapshot(raw)

            if row:
                records.append(row)

                if VERBOSE:
                    def fmt(val, decimals=1):
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            return f"{val:.{decimals}f}"
                        return "—"

                    t      = fmt(row.get("temperature"))
                    v_state = row.get("vib_intensity", "—")
                    p_int  = fmt(row.get("piezo_intensity"), 0)

                    r_v, r_i, r_p = (fmt(row.get("R_voltage")),
                                     fmt(row.get("R_current"), 2),
                                     fmt(row.get("R_power")))
                    r_f, r_pf = fmt(row.get("R_frequency")), fmt(row.get("R_pf"), 2)

                    s_v, s_i, s_p = (fmt(row.get("S_voltage")),
                                     fmt(row.get("S_current"), 2),
                                     fmt(row.get("S_power")))
                    s_f, s_pf = fmt(row.get("S_frequency")), fmt(row.get("S_pf"), 2)

                    t_v, t_i, t_p = (fmt(row.get("T_voltage")),
                                     fmt(row.get("T_current"), 2),
                                     fmt(row.get("T_power")))
                    t_f, t_pf = fmt(row.get("T_frequency")), fmt(row.get("T_pf"), 2)

                    print(f"✅ {row['timestamp']} | T={t}°C | "
                          f"Vib={v_state} (Piezo={p_int})")
                    print(f"   🔹 Phase R: {r_v} V | {r_i} A | {r_p} W | "
                          f"{r_f} Hz | PF: {r_pf}")
                    print(f"   🔹 Phase S: {s_v} V | {s_i} A | {s_p} W | "
                          f"{s_f} Hz | PF: {s_pf}")
                    print(f"   🔹 Phase T: {t_v} V | {t_i} A | {t_p} W | "
                          f"{t_f} Hz | PF: {t_pf}")
                    print("-" * 70)
            else:
                print("⚠️  No data returned from Firebase this cycle.")

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")

    if not records:
        print("❌ No records collected. Exiting without writing CSV.")
        return

    df = pd.DataFrame(records)

    expected_cols = [
        "timestamp", "timestamp_unix",
        "vib_count", "vib_intensity", "vibration_intensity",
        "piezo_intensity", "temperature",
        # aliases for downstream scripts
        "temperature_c", "vib_digital_count",
    ]
    for p in ("R", "S", "T"):
        expected_cols.extend([
            f"{p}_voltage", f"{p}_current", f"{p}_power",
            f"{p}_energy",  f"{p}_frequency", f"{p}_pf",
        ])

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols]
    df.to_csv(output, index=False)

    print(f"\n✅ Saved {len(df)} rows to {output}")
    print("📊 Columns collected:", df.columns.tolist())
    print("\n🔍 Non-Null Data Count:")
    print(df.notna().sum().to_string())


# ==============================================================
if __name__ == "__main__":
    try:
        init_firebase()
    except Exception as e:
        print(e)
        raise SystemExit(1)
    collect_to_csv()