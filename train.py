"""
train.py
Trains a fault-detection classifier on dataset_clean.csv
and saves:
    - fault_model.pkl
    - scaler.pkl
    - model_meta.json
    - reports/training_YYYYMMDD_HHMMSS/  ← all plots + text report

Plots generated (for research papers):
    01_class_distribution.png
    02_confusion_matrix.png
    03_feature_importance.png
    04_model_comparison.png
    05_feature_distributions.png
    06_correlation_heatmap.png
    07_roc_curve.png
    08_classification_report.png
    09_boxplots.png
    10_learning_curve.png
    training_report.txt
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("ℹ️  XGBoost not installed — using RandomForest only.")

# ============================================================
#  CONFIG
# ============================================================
CLEAN_FILE   = "dataset_clean.csv"
MODEL_FILE   = "fault_model.pkl"
SCALER_FILE  = "scaler.pkl"
META_FILE    = "model_meta.json"
REPORTS_DIR  = "reports"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

FEATURE_COLS = [
    "R_voltage", "S_voltage", "T_voltage",
    "R_current", "S_current", "T_current",
    "temperature_c",
    "vib_digital_count", "vib_intensity", "piezo_intensity",
]

NORMAL_RANGES = {
    "R_voltage":        (210, 250),
    "S_voltage":        (210, 250),
    "T_voltage":        (210, 250),
    "R_current":        (0, 15),
    "S_current":        (0, 15),
    "T_current":        (0, 15),
    "temperature_c":    (20, 45),
    "vib_digital_count": (0, 15),
    "vib_intensity":    (0, 2.0),
    "piezo_intensity":  (0, 4095),
}

# ============================================================
#  REPORT HELPERS
# ============================================================
def create_report_folder() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(REPORTS_DIR, f"training_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _save_fig(fig, folder: str, name: str):
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   📊 saved {path}")
    return path


# ============================================================
#  PLOT 01 — CLASS DISTRIBUTION (before / after augmentation)
# ============================================================
def plot_class_distribution(y_before, y_after, folder):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, y, title in [
        (axes[0], y_before, "Before Augmentation"),
        (axes[1], y_after,  "After Augmentation"),
    ]:
        unique, counts = np.unique(y, return_counts=True)
        labels = ["Normal" if u == 0 else "Fault" for u in unique]
        colors = ["#10b981" if u == 0 else "#ef4444" for u in unique]
        bars = ax.bar(labels, counts, color=colors, edgecolor="black")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Samples")
        ax.grid(axis="y", alpha=0.3)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(count), ha="center", va="bottom",
                    fontweight="bold")
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold")
    return _save_fig(fig, folder, "01_class_distribution.png")


# ============================================================
#  PLOT 02 — CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(y_test, y_pred, folder):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fault"])
    ax.set_yticklabels(["Normal", "Fault"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=18, fontweight="bold")
    fig.colorbar(im)
    return _save_fig(fig, folder, "02_confusion_matrix.png")


# ============================================================
#  PLOT 03 — FEATURE IMPORTANCE
# ============================================================
def plot_feature_importance(feature_names, importances, folder):
    order = np.argsort(importances)
    names = [feature_names[i] for i in order]
    vals  = importances[order]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(vals)))
    ax.barh(names, vals, color=colors, edgecolor="black")
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    return _save_fig(fig, folder, "03_feature_importance.png")


# ============================================================
#  PLOT 04 — MODEL COMPARISON
# ============================================================
def plot_model_comparison(results, folder):
    names = list(results.keys())
    cv   = [results[n]["cv"]   for n in names]
    test = [results[n]["test"] for n in names]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, cv,   w, label="CV Accuracy",   color="#3b82f6", edgecolor="black")
    ax.bar(x + w/2, test, w, label="Test Accuracy", color="#10b981", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for xi, (c, t) in zip(x, zip(cv, test)):
        ax.text(xi - w/2, c, f"{c:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, t, f"{t:.3f}", ha="center", va="bottom", fontsize=8)
    return _save_fig(fig, folder, "04_model_comparison.png")


# ============================================================
#  PLOT 05 — FEATURE DISTRIBUTIONS
# ============================================================
def plot_feature_distributions(df, folder):
    n = len(FEATURE_COLS)
    cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        normal = df[df["label"] == 0][col].dropna()
        fault  = df[df["label"] == 1][col].dropna()
        ax.hist(normal, bins=20, alpha=0.6, label="Normal", color="#10b981", edgecolor="black")
        ax.hist(fault,  bins=20, alpha=0.6, label="Fault",  color="#ef4444", edgecolor="black")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Distributions (Normal vs Fault)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, folder, "05_feature_distributions.png")


# ============================================================
#  PLOT 06 — CORRELATION HEATMAP
# ============================================================
def plot_correlation_heatmap(df, folder):
    corr = df[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURE_COLS, fontsize=8)
    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.colorbar(im)
    return _save_fig(fig, folder, "06_correlation_heatmap.png")


# ============================================================
#  PLOT 07 — ROC CURVE
# ============================================================
def plot_roc_curve(model, X_test, y_test, folder):
    try:
        if not hasattr(model, "predict_proba"):
            print("   ⚠ model has no predict_proba — skipping ROC")
            return None
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        return _save_fig(fig, folder, "07_roc_curve.png")
    except Exception as e:
        print(f"   ⚠ ROC curve failed: {e}")
        return None


# ============================================================
#  PLOT 08 — CLASSIFICATION REPORT
# ============================================================
def plot_classification_report(y_test, y_pred, folder):
    report = classification_report(y_test, y_pred,
                                   target_names=["Normal", "Fault"],
                                   output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    classes = ["Normal", "Fault"]
    data = np.array([[report[c][m] for m in metrics] for c in classes])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(metrics); ax.set_yticklabels(classes)
    ax.set_title("Classification Report", fontsize=13, fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if data[i, j] > 0.5 else "black")
    fig.colorbar(im)
    return _save_fig(fig, folder, "08_classification_report.png")


# ============================================================
#  PLOT 09 — BOX PLOTS
# ============================================================
def plot_boxplots(df, folder):
    n = len(FEATURE_COLS)
    cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        data = [df[df["label"] == 0][col].dropna(),
                df[df["label"] == 1][col].dropna()]
        bp = ax.boxplot(data, labels=["Normal", "Fault"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#10b981")
        bp["boxes"][1].set_facecolor("#ef4444")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Box Plots — Normal vs Fault", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, folder, "09_boxplots.png")


# ============================================================
#  PLOT 10 — LEARNING CURVE
# ============================================================
def plot_learning_curve(model, X_train, y_train, folder):
    try:
        sizes, tr, te = learning_curve(
            model, X_train, y_train, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy",
        )
        tm, ts = tr.mean(axis=1), te.mean(axis=1)
        tr_std, te_std = tr.std(axis=1), te.std(axis=1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(sizes, tm, "o-", color="#3b82f6", label="Training score")
        ax.fill_between(sizes, tm - tr_std, tm + tr_std, alpha=0.2, color="#3b82f6")
        ax.plot(sizes, ts, "o-", color="#10b981", label="CV score")
        ax.fill_between(sizes, ts - te_std, ts + te_std, alpha=0.2, color="#10b981")
        ax.set_xlabel("Training Examples"); ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)
        return _save_fig(fig, folder, "10_learning_curve.png")
    except Exception as e:
        print(f"   ⚠ Learning curve failed: {e}")
        return None


# ============================================================
#  TEXT REPORT
# ============================================================
def save_text_report(folder, best_name, best_score, results, meta,
                     n_before, n_after, dist_before, dist_after):
    path = os.path.join(folder, "training_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  FAULT DETECTION MODEL — TRAINING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date / Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset            : {CLEAN_FILE}\n")
        f.write(f"Rows (raw)         : {n_before}\n")
        f.write(f"Rows (augmented)   : {n_after}\n")
        f.write(f"Label dist (before): {dist_before}\n")
        f.write(f"Label dist (after) : {dist_after}\n\n")
        f.write(f"Best Model         : {best_name}\n")
        f.write(f"CV Accuracy        : {best_score:.4f}\n")
        f.write(f"Test Accuracy      : {results[best_name]['test']:.4f}\n\n")
        f.write("All models:\n")
        for name, r in results.items():
            f.write(f"  {name:15s}  CV={r['cv']:.4f}  Test={r['test']:.4f}\n")
        f.write("\nMeta JSON:\n")
        f.write(json.dumps(meta, indent=2))
    print(f"   📄 saved {path}")


# ============================================================
#  DATA LOADING / LABELING / AUGMENTATION
# ============================================================
def load_data(path=CLEAN_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run data_collect.py then data_clean.py first."
        )
    df = pd.read_csv(path)
    print(f"📂 Loaded {path}: shape={df.shape}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    df = df[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(how="all")
    print(f"   After all-NaN drop: {len(df)} rows (removed {before - len(df)})")
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill().fillna(0.0)
    return df.reset_index(drop=True)


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    fault = pd.Series(False, index=df.index)
    for col, (lo, hi) in NORMAL_RANGES.items():
        fault |= (df[col] < lo) | (df[col] > hi)
    df = df.copy()
    df["label"] = fault.astype(int)
    return df


def augment_if_needed(X, y, target_per_class=100, noise_frac=0.10):
    classes, counts = np.unique(y, return_counts=True)
    present = dict(zip(classes.tolist(), counts.tolist()))
    print(f"📊 Original label distribution: {present}")
    X_extra, y_extra = [], []

    if present.get(0, 0) < target_per_class:
        n = target_per_class - present.get(0, 0)
        print(f"   + Generating {n} synthetic NORMAL samples")
        for _ in range(n):
            X_extra.append([np.random.uniform(*NORMAL_RANGES[c]) for c in FEATURE_COLS])
            y_extra.append(0)

    if present.get(1, 0) < target_per_class:
        n = target_per_class - present.get(1, 0)
        print(f"   + Generating {n} synthetic FAULT samples")
        for _ in range(n):
            row = []
            for col in FEATURE_COLS:
                lo, hi = NORMAL_RANGES[col]
                span = hi - lo if hi > lo else 1.0
                v = (lo - np.random.uniform(0.1, 0.4) * span
                     if np.random.rand() < 0.5
                     else hi + np.random.uniform(0.1, 0.4) * span)
                row.append(v)
            X_extra.append(row)
            y_extra.append(1)

    if X_extra:
        X = np.vstack([X, np.array(X_extra)])
        y = np.concatenate([y, np.array(y_extra)])

    X = X + np.random.normal(0, noise_frac, X.shape) * (np.abs(X).mean(axis=0, keepdims=True) + 1e-6)
    classes, counts = np.unique(y, return_counts=True)
    print(f"📊 After augmentation: {dict(zip(classes.tolist(), counts.tolist()))}")
    return X, y


# ============================================================
#  MAIN TRAIN
# ============================================================
def train():
    report_dir = create_report_folder()
    print(f"\n📁 Report folder: {report_dir}\n")

    df = load_data()
    df = label_data(df)

    n_rows_before    = len(df)
    dist_before      = df["label"].value_counts().to_dict()

    print("📊 Generating data exploration plots …")
    plot_feature_distributions(df, report_dir)
    plot_correlation_heatmap(df, report_dir)
    plot_boxplots(df, report_dir)

    X = df[FEATURE_COLS].values
    y = df["label"].values
    y_before_aug = y.copy()

    X, y = augment_if_needed(X, y, target_per_class=100)
    dist_after = dict(zip(*np.unique(y, return_counts=True)))
    n_rows_after = len(y)

    plot_class_distribution(y_before_aug, y, report_dir)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Still only one class after augmentation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✂️  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    scaler   = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [50, 100], "max_depth": [5, 10, None],
             "min_samples_split": [2, 5]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss",
                          n_jobs=-1, use_label_encoder=False),
            {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1],
             "max_depth": [3, 5]},
        )

    best_model, best_name, best_score = None, "", -1.0
    results = {}
    for name, (clf, grid) in models.items():
        print(f"🔍 Training {name} …")
        gs = GridSearchCV(clf, grid, cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train_s, y_train)
        y_pred = gs.best_estimator_.predict(X_test_s)
        test_acc = accuracy_score(y_test, y_pred)
        results[name] = {"cv": gs.best_score_, "test": test_acc}
        print(f"   Best params  : {gs.best_params_}")
        print(f"   CV accuracy  : {gs.best_score_:.4f}")
        print(f"   Test accuracy: {test_acc:.4f}\n")
        if gs.best_score_ > best_score:
            best_score, best_model, best_name = gs.best_score_, gs.best_estimator_, name

    print("=" * 60)
    print(f"✅ Best model: {best_name} (CV acc = {best_score:.4f})")
    print("=" * 60)

    y_pred = best_model.predict(X_test_s)
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n",
          classification_report(y_test, y_pred,
                                target_names=["Normal", "Fault"], digits=4))

    print("\n📊 Generating result plots …")
    plot_confusion_matrix(y_test, y_pred, report_dir)
    plot_classification_report(y_test, y_pred, report_dir)
    plot_model_comparison(results, report_dir)
    plot_roc_curve(best_model, X_test_s, y_test, report_dir)
    plot_learning_curve(best_model, X_train_s, y_train, report_dir)

    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(FEATURE_COLS, best_model.feature_importances_, report_dir)
        print("\nFeature importance:")
        imp = sorted(zip(FEATURE_COLS, best_model.feature_importances_),
                     key=lambda kv: kv[1], reverse=True)
        for n, s in imp:
            print(f"   {n:<20s} {s:.4f}  {'█' * int(s * 40)}")

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    meta = {
        "model_name":    best_name,
        "feature_cols":  FEATURE_COLS,
        "normal_ranges": NORMAL_RANGES,
        "cv_accuracy":   best_score,
        "test_accuracy": results[best_name]["test"],
        "n_train":       int(len(X_train)),
        "n_test":        int(len(X_test)),
        "report_folder": report_dir,
        "trained_at":    datetime.now().isoformat(timespec="seconds"),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    save_text_report(report_dir, best_name, best_score, results, meta,
                     n_rows_before, n_rows_after, dist_before, dist_after)

    print(f"\n💾 Saved:\n   • {MODEL_FILE}\n   • {SCALER_FILE}\n   • {META_FILE}")
    print(f"   • All plots → {report_dir}")
    print("\n🎉 Done! Run:  python gui.py")


if __name__ == "__main__":
    print("=" * 60)
    print("  FAULT DETECTION MODEL TRAINING")
    print("=" * 60)
    try:
        train()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise