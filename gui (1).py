"""
gui.py — QUANTUM ENERGY MONITOR
"""

import csv
import json
import os
import smtplib
import sys
import time
import subprocess
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import joblib

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QProcess, QTimer, QPointF, QRectF, QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QFont, QTextCursor, QPainter, QColor, QPen, QBrush,
    QLinearGradient, QFontDatabase, QPainterPath
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QSizePolicy, QTabWidget, QPlainTextEdit,
    QSpinBox, QCheckBox, QGroupBox, QMessageBox, QGraphicsDropShadowEffect
)

from firebase_client import FirebaseClient

# ============================================================
#  THEMES (Day & Night)
# ============================================================
THEMES = {
    "day": {
        "bg_app":       "#f0f4f8",
        "bg_card":      "#ffffff",
        "bg_inner":     "#f8fafc",
        "border":       "#e2e8f0",
        "border_focus": "#3b82f6",
        "text_main":    "#1e293b",
        "text_sub":     "#64748b",
        "text_light":   "#94a3b8",
        "accent_blue":  "#2563eb",
        "accent_green": "#10b981",
        "accent_red":   "#ef4444",
        "accent_amber": "#f59e0b",
        "log_bg":       "#f8fafc",
        "log_text":     "#334155",
        "btn_bg":       "#ffffff",
        "btn_text":     "#334155",
        "btn_hover":    "#f1f5f9",
        "input_bg":     "#f8fafc",
        "input_border": "#cbd5e1",
        "alert_bg":     "#fef2f2",
        "alert_border": "#f87171",
    },
    "night": {
        "bg_app":       "#04060d",
        "bg_card":      "#0b1222",
        "bg_inner":     "#0a1420",
        "border":       "#1a2d4d",
        "border_focus": "#00d4ff",
        "text_main":    "#c8e8ff",
        "text_sub":     "#5a7a99",
        "text_light":   "#3a4a5d",
        "accent_blue":  "#00d4ff",
        "accent_green": "#00ff88",
        "accent_red":   "#ff3366",
        "accent_amber": "#ffaa00",
        "log_bg":       "#04060d",
        "log_text":     "#00ff88",
        "btn_bg":       "#0a1222",
        "btn_text":     "#c8e8ff",
        "btn_hover":    "#132038",
        "input_bg":     "#04060d",
        "input_border": "#1a2d4d",
        "alert_bg":     "#1a0812",
        "alert_border": "#ff3366",
    }
}

# ============================================================
#  CONFIG
# ============================================================
CSV_RAW     = "dataset.csv"
CSV_CLEAN   = "dataset_clean.csv"
MODEL_FILE  = "fault_model.pkl"
SCALER_FILE = "scaler.pkl"
META_FILE   = "model_meta.json"

LIVE_REFRESH_SEC          = 2
OFFLINE_AFTER_EMPTY_POLLS = 3

MEAN_DEV_WARN_PCT = 30
MEAN_DEV_CRIT_PCT = 50
MEAN_DEV_MIN_MEAN = 0.5

NORMAL_RANGES = {
    "R_voltage":         (210, 250),
    "S_voltage":         (210, 250),
    "T_voltage":         (210, 250),
    "R_current":         (0, 15),
    "S_current":         (0, 15),
    "T_current":         (0, 15),
    "temperature_c":     (20, 45),
    "vib_digital_count": (0, 15),
    "vib_intensity":     (0, 2.0),
    "piezo_intensity":   (0, 4095),
}

EMAIL_ENABLED = bool(os.environ.get("EMAIL_USER") and os.environ.get("EMAIL_PASSWORD"))
EMAIL_USER    = os.environ.get("EMAIL_USER", "")
EMAIL_PASS    = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_TO      = os.environ.get("EMAIL_TO", "sb284160@gmail.com")
EMAIL_SMTP    = "smtp.gmail.com"
EMAIL_PORT    = 587

VIB_LABEL_MAP = {
    "no vibration": 0.0, "no_vibration": 0.0, "none": 0.0,
    "low": 0.5, "medium": 1.0, "mid": 1.0, "high": 2.0,
}

# ============================================================
#  VOICE ENGINE
# ============================================================
class VoiceWorker(QThread):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", max(100, engine.getProperty("rate") - 20))
            engine.say(self.text)
            engine.runAndWait()
        except Exception as e:
            # Windows speech fallback. Base64 avoids quoting/injection issues.
            if os.name == "nt":
                try:
                    import base64
                    encoded = base64.b64encode(self.text.encode("utf-8")).decode("ascii")
                    ps = ("Add-Type -AssemblyName System.Speech; $b=[Convert]::FromBase64String('" + encoded + "'); "
                          "$t=[Text.Encoding]::UTF8.GetString($b); (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($t)")
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=30)
                except Exception as fallback_error:
                    print(f"TTS Error: {fallback_error}")
            else:
                print(f"TTS Error: {e}. Install pyttsx3 (pip install pyttsx3).")

# ============================================================
#  Helpers
# ============================================================
def vib_to_number(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    key = str(v).strip().lower()
    if key in VIB_LABEL_MAP:
        return VIB_LABEL_MAP[key]
    for label, num in VIB_LABEL_MAP.items():
        if key.startswith(label):
            return num
    return 0.0

def has_live_sensor_data(raw: dict) -> bool:
    """Return True if at least one sensor group contains a finite numeric reading.
    Zero is a valid reading. This is a presence check, not an anomaly check.
    """
    if not isinstance(raw, dict):
        return False
    groups = ("vibration", "temperature", "phaseR", "phaseS", "phaseT")
    for group_name in groups:
        group = raw.get(group_name)
        if not isinstance(group, dict):
            continue
        for value in group.values():
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
                if np.isfinite(number):
                    return True
            except (TypeError, ValueError):
                continue
    return False


CSV_HEADERS = [
    "timestamp", "timestamp_unix",
    "vib_count", "vib_intensity", "vibration_intensity",
    "piezo_intensity", "temperature", "temperature_c", "vib_digital_count",
    "R_voltage", "R_current", "R_power", "R_energy", "R_frequency", "R_pf",
    "S_voltage", "S_current", "S_power", "S_energy", "S_frequency", "S_pf",
    "T_voltage", "T_current", "T_power", "T_energy", "T_frequency", "T_pf",
]

def flatten_snapshot(raw: dict) -> dict:
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_unix": time.time(),
    }
    vib = raw.get("vibration") or {}
    row["vib_count"]           = vib.get("count")
    row["vib_intensity"]       = vib.get("intensity")
    row["piezo_intensity"]     = vib.get("piezoIntensity")
    row["vibration_intensity"] = row["piezo_intensity"]

    temp = raw.get("temperature") or {}
    row["temperature"] = temp.get("celsius")

    for phase in ("R", "S", "T"):
        p = raw.get(f"phase{phase}") or {}
        row[f"{phase}_voltage"]   = p.get("voltage")
        row[f"{phase}_current"]   = p.get("current")
        row[f"{phase}_power"]     = p.get("power")
        row[f"{phase}_energy"]    = p.get("energy")
        row[f"{phase}_frequency"] = p.get("frequency")
        row[f"{phase}_pf"]        = p.get("powerFactor")

    row["temperature_c"]     = row["temperature"]
    row["vib_digital_count"] = row["vib_count"]
    return row

def _fmt(val, decimals=1):
    if val is None:
        return "0.0" if decimals > 0 else "0"
    try:
        f = float(val)
        if np.isnan(f):
            return "0.0" if decimals > 0 else "0"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return "0.0" if decimals > 0 else "0"

def format_collected_row(row: dict) -> str:
    t = _fmt(row.get("temperature_c"), 1)
    vib = row.get("vib_intensity") or "—"
    piezo = _fmt(row.get("piezo_intensity"), 0)

    lines = [f"✅ {row['timestamp']} | T={t}°C | Vib={vib} (Piezo={piezo})"]
    for phase in ("R", "S", "T"):
        v  = _fmt(row.get(f"{phase}_voltage"),   1)
        i  = _fmt(row.get(f"{phase}_current"),   2)
        p  = _fmt(row.get(f"{phase}_power"),     1)
        f  = _fmt(row.get(f"{phase}_frequency"), 1)
        pf = _fmt(row.get(f"{phase}_pf"),        2)
        lines.append(
            f"   🔹 Phase {phase}: {v} V | {i} A | {p} W | {f} Hz | PF: {pf}"
        )
    lines.append("-" * 64)
    return "\n".join(lines)

def fingerprint_of(flat: dict) -> tuple:
    out = []
    for k in sorted(flat.keys()):
        if k in ("timestamp", "timestamp_unix"):
            continue
        v = flat.get(k)
        try:
            fv = float(v)
            out.append((k, None if np.isnan(fv) else round(fv, 3)))
        except (TypeError, ValueError):
            out.append((k, str(v) if v is not None else None))
    return tuple(out)

# ============================================================
#  UI COMPONENTS
# ============================================================
class CircularGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self._value = 0.0
        self._target = 0.0
        self._color = QColor("#2563eb")
        self._bg_color = QColor("#e2e8f0")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(30)

    def set_value(self, val, max_val=300):
        self._target = max(0.0, min(1.0, val / max_val))

    def set_colors(self, color, bg_color):
        self._color = QColor(color)
        self._bg_color = QColor(bg_color)

    def _animate(self):
        if abs(self._target - self._value) > 0.005:
            self._value += (self._target - self._value) * 0.1
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(10, 10, 100, 100)

        pen = QPen(self._bg_color, 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 0, 360 * 16)

        if self._value > 0:
            pen = QPen(self._color, 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            span = int(-self._value * 360 * 16)
            p.drawArc(rect, 90 * 16, span)

class PhaseCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("PhaseCard")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)

        self._is_crit = False
        self._is_flashing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        header = QHBoxLayout()
        lbl_title = QLabel(title.upper())
        lbl_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.addWidget(lbl_title)
        header.addStretch()
        lbl_icon = QLabel("◈")
        header.addWidget(lbl_icon)
        layout.addLayout(header)

        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(20)

        self.gauge = CircularGauge()
        mid_layout.addWidget(self.gauge)

        val_layout = QVBoxLayout()
        val_layout.setSpacing(5)
        val_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        volt_row = QHBoxLayout()
        self.lbl_volt = QLabel("0.00")
        self.lbl_volt.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.lbl_unit = QLabel("V")
        self.lbl_unit.setFont(QFont("Segoe UI", 14))
        self.lbl_unit.setAlignment(Qt.AlignmentFlag.AlignBottom)
        volt_row.addWidget(self.lbl_volt)
        volt_row.addWidget(self.lbl_unit)
        volt_row.addStretch()
        val_layout.addLayout(volt_row)

        self.lbl_status = QLabel("OFFLINE")
        self.lbl_status.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_status.setFixedWidth(80)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_layout.addWidget(self.lbl_status)
        val_layout.addStretch()

        mid_layout.addLayout(val_layout)
        layout.addLayout(mid_layout)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        layout.addWidget(line)

        grid = QGridLayout()
        grid.setSpacing(10)

        self.lbl_curr = self._make_metric(grid, 0, 0, "CURRENT")
        self.lbl_pow  = self._make_metric(grid, 0, 1, "POWER")
        self.lbl_freq = self._make_metric(grid, 1, 0, "FREQ")
        self.lbl_pf   = self._make_metric(grid, 1, 1, "PF")

        layout.addLayout(grid)

    def _make_metric(self, grid, row, col, title):
        box = QVBoxLayout()
        box.setSpacing(2)
        lbl_t = QLabel(title)
        lbl_t.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_v = QLabel("0.0")
        lbl_v.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))

        val_row = QHBoxLayout()
        val_row.addWidget(lbl_v)
        unit = "A" if "CURR" in title else "W" if "POW" in title else "Hz" if "FREQ" in title else ""
        if unit:
            lbl_u = QLabel(unit)
            lbl_u.setFont(QFont("Segoe UI", 9))
            val_row.addWidget(lbl_u)
        val_row.addStretch()

        box.addWidget(lbl_t)
        box.addLayout(val_row)

        wrapper = QWidget()
        wrapper.setLayout(box)
        grid.addWidget(wrapper, row, col)
        return lbl_v

    def update_data(self, volt, current, power, freq, pf, status, is_day):
        theme = THEMES["day" if is_day else "night"]

        if status == "offline" or volt is None:
            self.lbl_volt.setText("0.00")
            self.gauge.set_value(0)
            self.lbl_curr.setText("0.00")
            self.lbl_pow.setText("0.0")
            self.lbl_freq.setText("0.0")
            self.lbl_pf.setText("0.00")
            self.lbl_status.setText("OFFLINE")
            c = theme["text_sub"]
            self.lbl_status.setStyleSheet(f"color: {c}; background: {c}22; border-radius: 4px; padding: 2px 4px;")
            self.gauge.set_colors(c, theme["border"])
            return

        self.lbl_volt.setText(f"{float(volt):.2f}")
        self.gauge.set_value(float(volt))
        self.lbl_curr.setText(_fmt(current, 2))
        self.lbl_pow.setText(_fmt(power, 1))
        self.lbl_freq.setText(_fmt(freq, 1))
        self.lbl_pf.setText(_fmt(pf, 2))

        color_map = {
            "ok": theme["accent_green"],
            "warn": theme["accent_amber"],
            "crit": theme["accent_red"],
            "na": theme["text_sub"],
        }
        c = color_map.get(status, theme["accent_green"])

        self.lbl_status.setText(status.upper() if status != "ok" else "NOMINAL")
        self.lbl_status.setStyleSheet(f"color: {c}; background: {c}22; border-radius: 4px; padding: 2px 4px;")
        self.gauge.set_colors(c, theme["border"])

    def set_alert_flash(self, is_crit, flash_state, is_day):
        self._is_crit = is_crit
        self._is_flashing = flash_state
        theme = THEMES["day" if is_day else "night"]

        base_style = f"""
            QFrame#PhaseCard {{
                background: {theme['bg_card']};
                border-radius: 16px;
                border: 1px solid {theme['border']};
            }}
        """

        if self._is_crit and self._is_flashing:
            alert_style = f"""
                QFrame#PhaseCard {{
                    background: {theme['alert_bg']};
                    border-radius: 16px;
                    border: 2px solid {theme['alert_border']};
                }}
            """
            self.setStyleSheet(alert_style)
        else:
            self.setStyleSheet(base_style)

class SystemStatusCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SystemStatusCard")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        lbl_title = QLabel("SYSTEM STATUS")
        lbl_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(lbl_title)
        layout.addStretch()

        sub_grid = QHBoxLayout()
        sub_grid.setSpacing(10)

        self.card_temp = self._make_sub_card("TEMPERATURE", "0.0 °C")
        self.card_vib  = self._make_sub_card("VIBRATION", "NO VIBRATION")
        self.card_piezo = self._make_sub_card("PIEZO", "0")

        sub_grid.addWidget(self.card_temp)
        sub_grid.addWidget(self.card_vib)
        sub_grid.addWidget(self.card_piezo)

        layout.addLayout(sub_grid)

    def _make_sub_card(self, title, val):
        frame = QFrame()
        frame.setObjectName("SubCard")
        l = QVBoxLayout(frame)
        l.setContentsMargins(10, 10, 10, 10)
        l.setSpacing(5)

        lbl_t = QLabel(title)
        lbl_t.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_v = QLabel(val)
        lbl_v.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        lbl_v.setAlignment(Qt.AlignmentFlag.AlignCenter)

        l.addWidget(lbl_t)
        l.addWidget(lbl_v)
        frame.val_label = lbl_v
        return frame

    def update_data(self, temp, vib, piezo, is_day):
        if temp is None:
            self.card_temp.val_label.setText("0.0 °C")
            self.card_vib.val_label.setText("NO VIBRATION")
            self.card_piezo.val_label.setText("0")
            return

        self.card_temp.val_label.setText(f"{_fmt(temp, 1)} °C")
        self.card_vib.val_label.setText(str(vib).upper() if vib else "NO VIBRATION")
        self.card_piezo.val_label.setText(str(piezo) if piezo is not None else "0")

    def flash_metric(self, metric_name, is_crit, flash_state, is_day):
        theme = THEMES["day" if is_day else "night"]
        card_map = {"temp": self.card_temp, "vib": self.card_vib, "piezo": self.card_piezo}

        target_card = card_map.get(metric_name)
        if not target_card:
            return

        base_style = f"""
            QFrame#SubCard {{
                background: {theme['bg_inner']};
                border-radius: 10px;
                border: 1px solid {theme['border']};
            }}
        """

        if is_crit and flash_state:
            alert_style = f"""
                QFrame#SubCard {{
                    background: {theme['alert_bg']};
                    border-radius: 10px;
                    border: 2px solid {theme['alert_border']};
                }}
            """
            target_card.setStyleSheet(alert_style)
        else:
            target_card.setStyleSheet(base_style)

# ============================================================
#  WORKERS & DETECTOR
# ============================================================
class AnomalyDetector:
    def __init__(self):
        self.ranges = dict(NORMAL_RANGES)
        self.dataset_stats = {}
        self.model = None
        self.scaler = None
        self.feature_cols = []
        self.meta = {}
        self.reload()

    def reload(self):
        self._load_stats()
        self._load_model()

    def _load_stats(self):
        self.dataset_stats = {}
        if not os.path.exists(CSV_CLEAN):
            return
        try:
            df = pd.read_csv(CSV_CLEAN)
            for col in NORMAL_RANGES:
                if col not in df.columns:
                    continue
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) == 0:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                self.dataset_stats[col] = {
                    "min": float(s.min()), "max": float(s.max()),
                    "mean": float(s.mean()), "std": float(s.std(ddof=0)),
                    "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                    "lo_fence": float(q1 - 1.5 * iqr),
                    "hi_fence": float(q3 + 1.5 * iqr),
                    "n": int(len(s)),
                }
        except Exception as e:
            print(f"Stats load error: {e}")

    def _load_model(self):
        try:
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            with open(META_FILE) as f:
                self.meta = json.load(f)
            self.feature_cols = self.meta["feature_cols"]
        except Exception as e:
            self.model = self.scaler = None
            self.feature_cols = []

    def evaluate(self, flat: dict):
        per_key = {}
        triggered = []

        for key, (lo, hi) in self.ranges.items():
            val = flat.get(key)
            if val is None:
                per_key[key] = {"status": "offline", "reason": "missing", "dev_pct": None, "mean": None}
                continue
            try:
                v = vib_to_number(val) if key == "vib_intensity" else float(val)
            except (TypeError, ValueError):
                per_key[key] = {"status": "offline", "reason": "non-numeric", "dev_pct": None, "mean": None}
                continue
            if np.isnan(v):
                per_key[key] = {"status": "offline", "reason": "missing", "dev_pct": None, "mean": None}
                continue

            # Suppress alerts and mark as offline if the reading is exactly 0
            if v == 0.0:
                per_key[key] = {"status": "offline", "reason": "zero reading", "dev_pct": None, "mean": None}
                continue

            status, reason = "ok", ""
            dev_pct, mean = None, None
            span = hi - lo if hi > lo else 1.0

            if v < lo or v > hi:
                if v < lo - 0.2 * span or v > hi + 0.2 * span:
                    status = "crit"
                    reason = f"{key} = {v:.2f} critically outside {lo}–{hi}"
                else:
                    status = "warn"
                    reason = f"{key} = {v:.2f} outside normal {lo}–{hi}"
            else:
                st = self.dataset_stats.get(key)
                if st:
                    mean = st["mean"]
                    if abs(mean) >= MEAN_DEV_MIN_MEAN:
                        dev_pct = abs(v - mean) / abs(mean) * 100.0
                        if dev_pct >= MEAN_DEV_CRIT_PCT:
                            status = "crit"
                            reason = f"{key} = {v:.2f} is {dev_pct:.1f}% from avg {mean:.2f}"
                        elif dev_pct >= MEAN_DEV_WARN_PCT:
                            status = "warn"
                            reason = f"{key} = {v:.2f} is {dev_pct:.1f}% from avg {mean:.2f}"

                    if status == "ok" and st["std"] > 0:
                        z = abs(v - mean) / st["std"]
                        if z > 3:
                            status = "crit"
                            reason = f"{key} = {v:.2f} is {z:.1f}σ from avg {mean:.2f}"
                        elif v < st["lo_fence"] or v > st["hi_fence"]:
                            status = "warn"
                            reason = f"{key} = {v:.2f} outside IQR [{st['lo_fence']:.2f}, {st['hi_fence']:.2f}]"

            if status != "ok":
                triggered.append((key, reason, v, status))

            per_key[key] = {
                "status": status, "reason": reason,
                "dev_pct": dev_pct, "mean": mean,
            }

        ml = None
        if self.model is not None and self.scaler is not None and self.feature_cols:
            try:
                vec = []
                for col in self.feature_cols:
                    v = flat.get(col)
                    if col == "vib_intensity":
                        v = vib_to_number(v)
                    try:
                        vec.append(float(v) if v is not None else 0.0)
                    except (TypeError, ValueError):
                        vec.append(0.0)
                x = np.array(vec).reshape(1, -1)
                x_s = self.scaler.transform(x)
                pred = int(self.model.predict(x_s)[0])
                proba = (self.model.predict_proba(x_s)[0] if hasattr(self.model, "predict_proba") else [0.5, 0.5])
                ml = {"label": pred, "conf": float(proba[pred])}
            except Exception as e:
                print(f"ML predict error: {e}")

        return per_key, ml, triggered

class DataCollectWorker(QThread):
    row_written = pyqtSignal(dict)
    error       = pyqtSignal(str)
    finished_ok = pyqtSignal(int)

    def __init__(self, client: FirebaseClient, output: str, interval_sec: float):
        super().__init__()
        self.client = client
        self.output = output
        self.interval = max(0.2, interval_sec)
        self._running = False
        self._count = 0
        self._skip_count = 0

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        if not os.path.exists(self.output) or os.path.getsize(self.output) == 0:
            with open(self.output, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADERS)

        while self._running:
            try:
                snapshot = self.client.fetch_all_sensors() or {}
                row = flatten_snapshot(snapshot)

                # Presence check: a sensor reporting zero is still a valid reading.
                data_present = has_live_sensor_data(snapshot)

                if data_present:
                    with open(self.output, "a", newline="") as f:
                        csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)
                    self._count += 1
                    self.row_written.emit(row)
                else:
                    self._skip_count += 1
                    self.error.emit(f"All sensors zero/offline — row skipped ({self._skip_count} skipped)")
            except Exception as e:
                self.error.emit(str(e))

            chunks = max(1, int(self.interval * 10))
            for _ in range(chunks):
                if not self._running:
                    break
                self.msleep(100)

        self.finished_ok.emit(self._count)

class EmailWorker(QThread):
    sent  = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, subject: str, body: str):
        super().__init__()
        self.subject = subject
        self.body = body

    def run(self):
        try:
            msg = MIMEMultipart()
            msg["From"] = EMAIL_USER
            msg["To"] = EMAIL_TO
            msg["Subject"] = self.subject
            msg.attach(MIMEText(self.body, "plain"))
            with smtplib.SMTP(EMAIL_SMTP, EMAIL_PORT, timeout=15) as s:
                s.starttls()
                s.login(EMAIL_USER, EMAIL_PASS)
                s.send_message(msg)
            self.sent.emit(self.subject)
        except Exception as e:
            self.error.emit(str(e))

class LiveFetchWorker(QThread):
    data_ready = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, client: FirebaseClient, interval_sec: float):
        super().__init__()
        self.client = client
        self.interval = max(0.5, interval_sec)
        self._running = False

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                snapshot = self.client.fetch_all_sensors() or {}
                self.data_ready.emit(snapshot)
            except Exception as e:
                self.error.emit(str(e))
            chunks = max(1, int(self.interval * 10))
            for _ in range(chunks):
                if not self._running:
                    break
                self.msleep(100)

# ============================================================
#  DASHBOARD
# ============================================================
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QUANTUM ENERGY MONITOR")
        self.resize(1280, 850)

        try:
            self.fb = FirebaseClient()
        except Exception as e:
            QMessageBox.critical(None, "Firebase Error", str(e))
            raise

        self.detector = AnomalyDetector()
        self.is_day_mode = True
        self.flash_state = False
        self._is_speaking = False
        self.voice_enabled = True

        # State
        self._monitoring = False
        self._device_online = None
        self._consecutive_empty = 0
        self._last_valid_data_time = None
        self._voice_workers = []
        self.collector = None
        self.live_worker = None
        self.clean_proc = None
        self.train_proc = None
        self._sent_alerts = set()
        self._last_snapshot = None
        self._last_per_key = {}

        self._build_ui()
        self.apply_theme()

        # Animations
        self.flash_timer = QTimer(self)
        self.flash_timer.timeout.connect(self._toggle_flash)
        self.flash_timer.start(600)

        self._start_monitoring()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        # ===================== HEADER =====================
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(80)

        shadow = QGraphicsDropShadowEffect(header)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 5)
        header.setGraphicsEffect(shadow)

        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(24, 10, 24, 10)

        title_box = QVBoxLayout()
        self.title_label = QLabel("◈ QUANTUM ENERGY MONITOR")
        self.title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        subtitle = QLabel("REAL-TIME · MULTI-LAYER · AI-POWERED")
        subtitle.setFont(QFont("Segoe UI", 8))
        title_box.addWidget(self.title_label)
        title_box.addWidget(subtitle)
        h_layout.addLayout(title_box)

        h_layout.addStretch()

        # Status Pill with Breathing Animation
        self.status_pill = QFrame()
        self.status_pill.setObjectName("StatusPill")
        sp_layout = QHBoxLayout(self.status_pill)
        sp_layout.setContentsMargins(15, 5, 15, 5)

        self.header_dot = QLabel("●")
        self.header_dot.setFont(QFont("Segoe UI", 12))
        self.conn_label = QLabel("ONLINE")
        self.conn_label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        sp_layout.addWidget(self.header_dot)
        sp_layout.addWidget(self.conn_label)
        h_layout.addWidget(self.status_pill)

        # Setup Pulse Animation for Status Pill
        self.pulse_anim = QPropertyAnimation(self.status_pill, b"styleSheet")
        self.pulse_anim.setDuration(1500)
        self.pulse_anim.setLoopCount(-1)
        self._start_pulse()

        # Voice Toggle
        self.voice_btn = QPushButton("🔊")
        self.voice_btn.setFixedSize(40, 40)
        self.voice_btn.setFont(QFont("Segoe UI", 12))
        self.voice_btn.setToolTip("Toggle Voice Alerts")
        self.voice_btn.clicked.connect(self._toggle_voice)
        h_layout.addWidget(self.voice_btn)

        # Theme Toggle
        self.theme_btn = QPushButton("☀️ / 🌙")
        self.theme_btn.setFixedSize(50, 40)
        self.theme_btn.setFont(QFont("Segoe UI", 12))
        self.theme_btn.clicked.connect(self._toggle_theme)
        h_layout.addWidget(self.theme_btn)

        root.addWidget(header)

        # ===================== TABS =====================
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_live_tab(), "LIVE DASHBOARD")
        self.tabs.addTab(self._build_data_tab(), "DATA & TRAINING")
        root.addWidget(self.tabs, 1)

    def _start_pulse(self):
        t = THEMES["day" if self.is_day_mode else "night"]
        c = t['accent_green']
        pulse_css = f"""
            QFrame#StatusPill {{
                background: {c}22;
                border-radius: 20px;
                border: 2px solid {c};
            }}
        """
        self.pulse_anim.setStartValue(self.status_pill.styleSheet())
        self.pulse_anim.setEndValue(pulse_css)
        self.pulse_anim.start()

    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top Control Bar
        control_bar = QHBoxLayout()

        self.monitor_btn = QPushButton("▶ START MONITORING")
        self.monitor_btn.setFixedHeight(40)
        self.monitor_btn.setFixedWidth(180)
        self.monitor_btn.clicked.connect(self._toggle_monitoring)
        control_bar.addWidget(self.monitor_btn)

        self.btn_glow = QGraphicsDropShadowEffect(self.monitor_btn)
        self.btn_glow.setBlurRadius(15)
        self.btn_glow.setOffset(0, 0)
        self.btn_glow.setColor(QColor(0, 0, 0, 0))
        self.monitor_btn.setGraphicsEffect(self.btn_glow)

        control_bar.addStretch()
        layout.addLayout(control_bar)

        # Top Row: 3 Phase Cards
        top_row = QHBoxLayout()
        top_row.setSpacing(20)

        self.card_R = PhaseCard("Phase R")
        self.card_S = PhaseCard("Phase S")
        self.card_T = PhaseCard("Phase T")

        top_row.addWidget(self.card_R)
        top_row.addWidget(self.card_S)
        top_row.addWidget(self.card_T)

        layout.addLayout(top_row, 4)

        # Bottom Row: Status + Log
        bot_row = QHBoxLayout()
        bot_row.setSpacing(20)

        self.status_card = SystemStatusCard()
        bot_row.addWidget(self.status_card, 2)

        log_frame = QFrame()
        log_frame.setObjectName("Card")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(20, 20, 20, 20)

        log_title = QLabel("LIVE ALERT LOG")
        log_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        log_layout.addWidget(log_title)

        self.alert_log = QPlainTextEdit()
        self.alert_log.setReadOnly(True)
        self.alert_log.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.alert_log)

        bot_row.addWidget(log_frame, 5)

        layout.addLayout(bot_row, 3)

        return w

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        self.collect_box = QGroupBox("DATA COLLECTION")
        c_layout = QHBoxLayout(self.collect_box)
        c_layout.setSpacing(15)

        c_layout.addWidget(QLabel("DURATION (min):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(0, 600)
        self.duration_spin.setValue(5)
        c_layout.addWidget(self.duration_spin)

        c_layout.addWidget(QLabel("INTERVAL (s):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(1)
        c_layout.addWidget(self.interval_spin)

        self.start_collect_btn = QPushButton("▶ START COLLECTION")
        self.start_collect_btn.clicked.connect(self._start_collection)
        c_layout.addWidget(self.start_collect_btn)

        self.stop_collect_btn = QPushButton("⏹ STOP")
        self.stop_collect_btn.setEnabled(False)
        self.stop_collect_btn.clicked.connect(self._stop_collection)
        c_layout.addWidget(self.stop_collect_btn)

        c_layout.addStretch()
        layout.addWidget(self.collect_box)

        self.train_box = QGroupBox("CLEAN & TRAIN")
        t_layout = QHBoxLayout(self.train_box)
        t_layout.setSpacing(15)

        self.clean_btn = QPushButton("🧹 CLEAN CSV")
        self.clean_btn.clicked.connect(self._run_clean)
        t_layout.addWidget(self.clean_btn)

        self.train_btn = QPushButton("🎯 TRAIN MODEL")
        self.train_btn.clicked.connect(self._run_train)
        t_layout.addWidget(self.train_btn)

        self.pipeline_btn = QPushButton("🚀 CLEAN + TRAIN")
        self.pipeline_btn.clicked.connect(self._run_clean_then_train)
        t_layout.addWidget(self.pipeline_btn)

        self.reload_btn = QPushButton("🔄 RELOAD MODEL")
        self.reload_btn.clicked.connect(self._reload_model)
        t_layout.addWidget(self.reload_btn)

        t_layout.addStretch()
        layout.addWidget(self.train_box)

        self.auto_chain_cb = QCheckBox("AUTO CLEAN + TRAIN ON STOP")
        self.auto_chain_cb.setChecked(True)
        layout.addWidget(self.auto_chain_cb)

        layout.addWidget(QLabel("COLLECTION LOG"))
        self.process_log = QPlainTextEdit()
        self.process_log.setReadOnly(True)
        self.process_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.process_log, 1)

        return w

    # ============================================================
    #  THEME & SETTINGS HANDLING
    # ============================================================
    def _toggle_theme(self):
        self.is_day_mode = not self.is_day_mode
        self.apply_theme()

    def _toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        if self.voice_enabled:
            self.voice_btn.setText("🔊")
            self.voice_btn.setStyleSheet("background: #10b981; color: white; border-radius: 8px;")
        else:
            self.voice_btn.setText("🔇")
            self.voice_btn.setStyleSheet("background: #ef4444; color: white; border-radius: 8px;")

    def apply_theme(self):
        t = THEMES["day" if self.is_day_mode else "night"]

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {t['bg_app']};
                color: {t['text_main']};
                font-family: "Segoe UI", sans-serif;
            }}
            QFrame#Header {{
                background: {t['bg_card']};
                border-radius: 12px;
                border: 1px solid {t['border']};
            }}
            QFrame#StatusPill {{
                background: {t['bg_inner']};
                border-radius: 20px;
                border: 1px solid {t['border']};
            }}
            QFrame#PhaseCard, QFrame#SystemStatusCard, QFrame#Card {{
                background: {t['bg_card']};
                border-radius: 16px;
                border: 1px solid {t['border']};
            }}
            QFrame#SubCard {{
                background: {t['bg_inner']};
                border-radius: 10px;
                border: 1px solid {t['border']};
            }}
            QLabel {{
                color: {t['text_main']};
            }}
            QTabWidget::pane {{
                border: none;
                background: transparent;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {t['text_sub']};
                padding: 10px 20px;
                font-weight: bold;
                font-size: 11px;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {t['accent_blue']};
                border-bottom: 2px solid {t['accent_blue']};
            }}
            QPlainTextEdit {{
                background: {t['log_bg']};
                color: {t['log_text']};
                border: 1px solid {t['border']};
                border-radius: 8px;
                padding: 10px;
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {t['border']};
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background: {t['bg_card']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {t['accent_blue']};
            }}
            QPushButton {{
                background: {t['btn_bg']};
                color: {t['btn_text']};
                border: 1px solid {t['border']};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {t['btn_hover']};
            }}
            QPushButton:disabled {{
                color: {t['text_light']};
            }}
            QSpinBox {{
                background: {t['input_bg']};
                color: {t['text_main']};
                border: 1px solid {t['input_border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QCheckBox {{
                color: {t['text_main']};
            }}
        """)

        self.title_label.setStyleSheet(f"color: {t['accent_blue']};")
        self.conn_label.setStyleSheet(f"color: {t['accent_green']};")
        self.header_dot.setStyleSheet(f"color: {t['accent_green']};")
        self.theme_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['bg_inner']};
                border: 1px solid {t['border']};
                border-radius: 8px;
            }}
            QPushButton:hover {{ background: {t['btn_hover']}; }}
        """)

        if self.voice_enabled:
            self.voice_btn.setStyleSheet("background: #10b981; color: white; border-radius: 8px;")
        else:
            self.voice_btn.setStyleSheet("background: #ef4444; color: white; border-radius: 8px;")

        self._style_monitor_btn()

        self.card_R.gauge.set_colors(t['accent_blue'], t['border'])
        self.card_S.gauge.set_colors(t['accent_blue'], t['border'])
        self.card_T.gauge.set_colors(t['accent_blue'], t['border'])

        self._start_pulse()
        self._apply_all_flashes()

    def _style_monitor_btn(self):
        t = THEMES["day" if self.is_day_mode else "night"]
        if self._monitoring:
            self.monitor_btn.setText("⏹ STOP MONITORING")
            bg = t['alert_bg']
            fg = t['accent_red']
            border = t['accent_red']
            self.btn_glow.setColor(QColor(fg))
        else:
            self.monitor_btn.setText("▶ START MONITORING")
            bg = t['bg_card']
            fg = t['accent_green']
            border = t['accent_green']
            self.btn_glow.setColor(QColor(0, 0, 0, 0))

        self.monitor_btn.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {fg};
                border: 2px solid {border};
                border-radius: 8px;
                font-size: 11px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {border}33;
            }}
        """)

    # ============================================================
    #  FLASH ANIMATION LOGIC
    # ============================================================
    def _toggle_flash(self):
        self.flash_state = not self.flash_state
        self._apply_all_flashes()

    def _apply_all_flashes(self):
        if not self._last_per_key:
            return

        for phase, card in [("R", self.card_R), ("S", self.card_S), ("T", self.card_T)]:
            keys = [f"{phase}_voltage", f"{phase}_current", f"{phase}_power", f"{phase}_frequency", f"{phase}_pf"]
            is_crit = any(self._last_per_key.get(k, {}).get("status") == "crit" for k in keys)
            card.set_alert_flash(is_crit, self.flash_state, self.is_day_mode)

        self.status_card.flash_metric("temp", self._last_per_key.get("temperature_c", {}).get("status") == "crit", self.flash_state, self.is_day_mode)
        self.status_card.flash_metric("vib", self._last_per_key.get("vib_intensity", {}).get("status") == "crit", self.flash_state, self.is_day_mode)
        self.status_card.flash_metric("piezo", self._last_per_key.get("piezo_intensity", {}).get("status") == "crit", self.flash_state, self.is_day_mode)

    # ============================================================
    #  VOICE & RECOMMENDATIONS
    # ============================================================
    def _speak_alert(self, triggered):
        if not self.voice_enabled or self._is_speaking or not triggered:
            return

        crit = [t for t in triggered if t[3] == "crit"]
        target = crit[0] if crit else triggered[0]

        key, reason, value, severity = target

        recommendation = ""
        if "voltage" in key:
            phase = key.split("_")[0]
            recommendation = f"Please check the Phase {phase} voltage connection."
        elif "current" in key:
            phase = key.split("_")[0]
            recommendation = f"Please check the Phase {phase} load."
        elif "temperature" in key:
            recommendation = "Please check the cooling system or ambient temperature."
        elif "vib" in key or "piezo" in key:
            recommendation = "Please inspect the machine for mechanical faults."
        else:
            recommendation = "Please check the sensor connection."

        prefix = "Critical alert. " if severity == "crit" else "Warning. "
        text = f"{prefix}{reason.replace('=', 'equals')}. {recommendation}"

        self._is_speaking = True
        self.voice_worker = VoiceWorker(text)
        self.voice_worker.finished.connect(lambda: setattr(self, '_is_speaking', False))
        self.voice_worker.start()

    # ============================================================
    #  MONITORING & REAL-TIME LOGIC
    # ============================================================
    def _toggle_monitoring(self):
        if self._monitoring:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _announce_device_status(self, online: bool):
        if not self.voice_enabled:
            return
        text = "Quantum Energy Monitor device is online." if online else "Quantum Energy Monitor device is offline."
        worker = VoiceWorker(text)
        self._voice_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._voice_workers.remove(w) if w in self._voice_workers else None)
        worker.start()

    def _start_monitoring(self):
        if self._monitoring:
            return
        self._monitoring = True
        self._device_online = None
        self._consecutive_empty = 0
        self._last_valid_data_time = None
        self._last_per_key = {}
        self._sent_alerts.clear()
        self.live_worker = LiveFetchWorker(self.fb, LIVE_REFRESH_SEC)
        self.live_worker.data_ready.connect(self._on_live_data)
        self.live_worker.error.connect(self._on_fetch_error)
        self.live_worker.start()
        self._style_monitor_btn()
        self._log("Monitoring started; waiting for sensor data.")

    def _stop_monitoring(self):
        if not self._monitoring:
            return
        self._monitoring = False
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker.wait(2000)
            self.live_worker = None
        self._device_online = None
        self._sent_alerts.clear()
        self._last_per_key = {}
        self.card_R.update_data(None, None, None, None, None, "offline", self.is_day_mode)
        self.card_S.update_data(None, None, None, None, None, "offline", self.is_day_mode)
        self.card_T.update_data(None, None, None, None, None, "offline", self.is_day_mode)
        self.status_card.update_data(None, None, None, self.is_day_mode)
        self._apply_all_flashes()
        t = THEMES["day" if self.is_day_mode else "night"]
        self.header_dot.setStyleSheet(f"color: {t['text_sub']};")
        self.conn_label.setText("STOPPED")
        self.conn_label.setStyleSheet(f"color: {t['text_sub']};")
        self._style_monitor_btn()
        self._log("Monitoring stopped.")

    def _on_fetch_error(self, msg: str):
        if not self._monitoring:
            return
        self._consecutive_empty += 1
        self._log(f"Firebase fetch error ({self._consecutive_empty}/{OFFLINE_AFTER_EMPTY_POLLS}): {msg}",
                  THEMES["day" if self.is_day_mode else "night"]["accent_amber"])
        if self._consecutive_empty >= OFFLINE_AFTER_EMPTY_POLLS:
            self._set_offline("Firebase fetch errors")

    def _set_online(self):
        if self._device_online is True:
            return
        t = THEMES["day" if self.is_day_mode else "night"]
        self._device_online = True
        self.header_dot.setStyleSheet(f"color: {t['accent_green']};")
        self.conn_label.setText("ONLINE")
        self.conn_label.setStyleSheet(f"color: {t['accent_green']};")
        self._log("DEVICE ONLINE", t["accent_green"])
        self._announce_device_status(True)

    def _set_offline(self, reason: str = ""):
        if self._device_online is False:
            return
        t = THEMES["day" if self.is_day_mode else "night"]
        self._device_online = False
        self.card_R.update_data(None, None, None, None, None, "offline", self.is_day_mode)
        self.card_S.update_data(None, None, None, None, None, "offline", self.is_day_mode)
        self.card_T.update_data(None, None, None, None, None, "offline", self.is_day_mode)
        self.status_card.update_data(None, None, None, self.is_day_mode)
        self._last_per_key = {}
        self._apply_all_flashes()
        self.header_dot.setStyleSheet(f"color: {t['accent_red']};")
        self.conn_label.setText("OFFLINE")
        self.conn_label.setStyleSheet(f"color: {t['accent_red']};")
        self._log(f"DEVICE OFFLINE: {reason}", t["accent_red"])
        self._announce_device_status(False)

    def _on_live_data(self, snapshot: dict):
        if not self._monitoring:
            return
        if not isinstance(snapshot, dict) or not has_live_sensor_data(snapshot):
            self._consecutive_empty += 1
            if self._consecutive_empty >= OFFLINE_AFTER_EMPTY_POLLS:
                self._set_offline("No sensor data received")
            return

        # Any valid reading, including zero, establishes device connectivity.
        self._consecutive_empty = 0
        self._last_valid_data_time = time.time()
        if self._device_online is not True:
            self._set_online()

        flat = flatten_snapshot(snapshot)
        per_key, ml, triggered = self.detector.evaluate(flat)
        self._last_per_key = per_key
        t = THEMES["day" if self.is_day_mode else "night"]

        for phase, card in [("R", self.card_R), ("S", self.card_S), ("T", self.card_T)]:
            v = flat.get(f"{phase}_voltage")
            c = flat.get(f"{phase}_current")
            p = flat.get(f"{phase}_power")
            f = flat.get(f"{phase}_frequency")
            pf = flat.get(f"{phase}_pf")
            keys = [f"{phase}_voltage", f"{phase}_current", f"{phase}_power", f"{phase}_frequency", f"{phase}_pf"]
            statuses = [per_key.get(k, {}).get("status", "offline") for k in keys]
            if "crit" in statuses:
                status = "crit"
            elif "warn" in statuses:
                status = "warn"
            elif all(s == "offline" for s in statuses):
                status = "offline"
            else:
                status = "ok"
            card.update_data(v, c, p, f, pf, status, self.is_day_mode)

        self.status_card.update_data(flat.get("temperature_c"), flat.get("vib_intensity"),
                                     flat.get("piezo_intensity"), self.is_day_mode)
        self._apply_all_flashes()

        if triggered:
            self._speak_alert(triggered)
        for key, reason, value, severity in triggered:
            icon = "✖" if severity == "crit" else "⚠"
            color = t["accent_red"] if severity == "crit" else t["accent_amber"]
            self._log(f"{icon} {reason}", color)
            alert_key = (key, severity)
            if severity == "crit" and alert_key not in self._sent_alerts:
                self._send_alert_email(key, reason, value)
                self._sent_alerts.add(alert_key)
        active = {(k, sev) for k, _, _, sev in triggered}
        self._sent_alerts &= active

        if ml and ml["label"] == 1:
            alert_key = ("__ml__", "crit")
            if alert_key not in self._sent_alerts:
                msg = f"ML model predicts FAULT (confidence {ml['conf']:.1%})"
                self._log(f"🚨 {msg}", t["accent_red"])
                self._send_alert_email("ML model", msg, None)
                self._sent_alerts.add(alert_key)

    def _log(self, message: str, color: str = None):
        ts = datetime.now().strftime("%H:%M:%S")
        if not color:
            color = THEMES["day" if self.is_day_mode else "night"]["log_text"]
        self.alert_log.appendHtml(f"<span style='color: #94a3b8;'>{ts}</span> <span style='color: {color};'>{message}</span>")
        self.alert_log.moveCursor(QTextCursor.MoveOperation.End)

    def _log_proc(self, message: str):
        self.process_log.appendPlainText(message)
        self.process_log.moveCursor(QTextCursor.MoveOperation.End)

    def _send_alert_email(self, sensor: str, reason: str, value=None):
        if not EMAIL_ENABLED:
            return
        subject = f"🚨 Sensor Alert: {sensor}"
        body = f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nSensor    : {sensor}\nReason    : {reason}\nValue     : {value}"
        worker = EmailWorker(subject, body)
        worker.sent.connect(lambda s: self._log(f"📧 EMAIL SENT: {s}"))
        worker.error.connect(lambda e: self._log(f"📧 EMAIL ERR: {e}"))
        worker.start()

    # ============================================================
    #  DATA COLLECTION & TRAINING
    # ============================================================
    def _start_collection(self):
        if self.collector and self.collector.isRunning():
            return
        interval = float(self.interval_spin.value())
        self.collector = DataCollectWorker(self.fb, CSV_RAW, interval)
        self.collector.row_written.connect(lambda row: self._log_proc(format_collected_row(row)))
        self.collector.error.connect(lambda e: self._log_proc(f"⚠ {e}"))
        self.collector.finished_ok.connect(self._on_collection_finished)
        self.collector.start()

        self.start_collect_btn.setEnabled(False)
        self.stop_collect_btn.setEnabled(True)
        self._log_proc("▶ COLLECTION STARTED")

        minutes = self.duration_spin.value()
        if minutes > 0:
            self._collect_timer = QTimer(self)
            self._collect_timer.setSingleShot(True)
            self._collect_timer.timeout.connect(self._stop_collection)
            self._collect_timer.start(minutes * 60_000)

    def _stop_collection(self):
        if self.collector and self.collector.isRunning():
            self.collector.stop()

    def _on_collection_finished(self, count: int):
        self.start_collect_btn.setEnabled(True)
        self.stop_collect_btn.setEnabled(False)
        self._log_proc(f"✅ COLLECTION FINISHED — {count} rows written")

        if self.auto_chain_cb.isChecked() and count > 0:
            self._run_clean_then_train()

    def _run_qprocess(self, script: str, label: str) -> QProcess:
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda p=proc: self._log_proc(bytes(p.readAllStandardOutput()).decode(errors="replace").rstrip())
        )
        proc.finished.connect(lambda code, status: self._log_proc(f"── {label.upper()} FINISHED (exit {code}) ──"))
        proc.start(sys.executable, [script])
        self._log_proc(f"▶ RUNNING {script} …")
        return proc

    def _run_clean(self):
        self.clean_proc = self._run_qprocess("data_clean.py", "clean")

    def _run_train(self):
        self.train_proc = self._run_qprocess("train.py", "train")
        self.train_proc.finished.connect(self._on_train_finished)

    def _run_clean_then_train(self):
        self.clean_proc = self._run_qprocess("data_clean.py", "clean")
        self.clean_proc.finished.connect(lambda code, _: self._run_train() if code == 0 else self._log_proc("✖ Clean failed."))

    def _on_train_finished(self, code, _status):
        if code == 0:
            self._reload_model()

    def _reload_model(self):
        self.detector.reload()
        self._log_proc(f"🔄 Model reloaded.")

    def closeEvent(self, event):
        self._monitoring = False
        if self.collector and self.collector.isRunning():
            self.collector.stop()
            self.collector.wait(2000)
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.stop()
            self.live_worker.wait(2000)
        event.accept()

# ============================================================
def main():
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()