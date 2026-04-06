#!/usr/bin/env python3
"""
model.py — Breach Intel ML Scoring Engine
RandomForest (300 trees, 4-class) + IsolationForest + SHAP TreeExplainer
Reads normalised feature vector from stdin (JSON).
Outputs { score, risk_level, factors, shap_factors } to stdout (JSON).
HARD RULE: all-zero feature vector → score 0 / NO EXPOSURE (never use model).
"""

import sys
import json
import os
import math
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

MODEL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "breach_rf_model.joblib")
ISO_PATH   = os.path.join(MODEL_DIR, "isolation_forest.joblib")

FEATURE_NAMES = [
    "breach_norm", "password_norm", "severity_norm", "critical_norm",
    "recent_norm", "login_anomaly_score", "public_exposure",
    "social_risk_score", "has_password_breach",
]

# 4-class calibrated centroids — aligned to real breach risk tiers
# LOW    (1-3 breaches, email-only, low sev)   →  8-24
# MEDIUM (2-8 breaches, some passwords)        → 25-49
# HIGH   (5-20 breaches, passwords, high sev)  → 50-74
# CRITICAL (15+ breaches, critical)            → 75-100
CENTROIDS = {"low": 15, "medium": 37, "high": 62, "critical": 85}


def is_zero_vector(feature_vec):
    """
    Return True if breach_norm is zero OR all primary signals are zero.
    breach_norm = 0 means breach_count = 0 → no exposure, period.
    Even if other signals are non-zero (malformed input), no breach = no risk.
    """
    # Primary rule: no breach count at all
    if feature_vec[0] == 0.0:  # breach_norm
        return True
    # Secondary: all key signals zero
    key_indices = [0, 1, 2, 3, 8]
    return all(feature_vec[i] == 0.0 for i in key_indices)


def zero_result():
    return {
        "score": 0,
        "risk_level": "NO EXPOSURE",
        "factors": [
            {"icon": "🔑", "name": "PASSWORD LEAKS",  "score": 0, "barColor": "var(--accent-red)"},
            {"icon": "💀", "name": "BREACH SEVERITY", "score": 0, "barColor": "var(--accent-orange)"},
            {"icon": "🔁", "name": "EXPOSURE COUNT",  "score": 0, "barColor": "var(--accent-yellow)"},
            {"icon": "⚡", "name": "RECENT BREACHES", "score": 0, "barColor": "var(--accent-cyan)"},
            {"icon": "🌐", "name": "PUBLIC EXPOSURE", "score": 0, "barColor": "var(--accent-blue)"},
        ],
        "shap_factors": [],
        "proba": {"low": 1.0, "medium": 0.0, "high": 0.0, "critical": 0.0},
    }


def generate_training_data():
    """
    4-class realistic training data.
    breach_norm cap = 60: log(2)/log(61)=0.13, log(4)/log(61)=0.27,
                          log(6)/log(61)=0.42, log(21)/log(61)=0.73, log(61)/log(61)=1.0
    """
    np.random.seed(42)
    X, y = [], []

    # LOW: 1-3 breaches, mostly email-only, low severity
    for _ in range(150):
        row = [
            np.random.uniform(0.05, 0.28),  # breach: 1-3
            np.random.uniform(0.00, 0.08),  # password: minimal
            np.random.uniform(0.10, 0.45),  # severity: low-mid
            0.0,                            # critical: none
            np.random.uniform(0.00, 0.15),  # recent: rare
            np.random.uniform(0.01, 0.25),  # login anomaly
            np.random.uniform(0.01, 0.10),  # public exposure
            np.random.uniform(0.01, 0.20),  # social risk
            0.0,                            # no password breach
        ]
        X.append(row); y.append("low")

    # MEDIUM: 2-8 breaches, some passwords, medium severity
    for _ in range(150):
        row = [
            np.random.uniform(0.20, 0.52),  # breach: 2-8
            np.random.uniform(0.08, 0.40),  # password: some
            np.random.uniform(0.35, 0.68),  # severity: medium
            np.random.uniform(0.00, 0.18),  # critical: rare
            np.random.uniform(0.10, 0.45),  # recent: some
            np.random.uniform(0.20, 0.60),  # login anomaly
            np.random.uniform(0.08, 0.35),  # public exposure
            np.random.uniform(0.20, 0.55),  # social risk
            float(np.random.choice([0, 1], p=[0.35, 0.65])),
        ]
        X.append(row); y.append("medium")

    # HIGH: 5-20 breaches, multiple passwords, high severity
    for _ in range(120):
        row = [
            np.random.uniform(0.40, 0.73),  # breach: 5-20
            np.random.uniform(0.35, 0.70),  # password: many
            np.random.uniform(0.60, 0.85),  # severity: high
            np.random.uniform(0.10, 0.45),  # critical: some
            np.random.uniform(0.30, 0.70),  # recent: many
            np.random.uniform(0.50, 0.85),  # login anomaly
            np.random.uniform(0.30, 0.65),  # public exposure
            np.random.uniform(0.50, 0.80),  # social risk
            1.0,
        ]
        X.append(row); y.append("high")

    # CRITICAL: 15+ breaches, critical entries, max severity
    for _ in range(100):
        row = [
            np.random.uniform(0.68, 1.00),  # breach: 15+
            np.random.uniform(0.60, 1.00),  # password: very many
            np.random.uniform(0.80, 1.00),  # severity: critical
            np.random.uniform(0.35, 1.00),  # critical: lots
            np.random.uniform(0.50, 1.00),  # recent: very many
            np.random.uniform(0.70, 1.00),  # login anomaly
            np.random.uniform(0.55, 1.00),  # public exposure
            np.random.uniform(0.65, 1.00),  # social risk
            1.0,
        ]
        X.append(row); y.append("critical")

    return np.array(X, dtype=np.float32), y


def get_or_train_models(force_retrain=False):
    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(ISO_PATH):
        try:
            rf  = joblib.load(MODEL_PATH)
            iso = joblib.load(ISO_PATH)
            # Must be 4-class model
            if hasattr(rf, 'classes_') and len(rf.classes_) == 4:
                return rf, iso
        except Exception:
            pass

    X, y = generate_training_data()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X, y)
    iso = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    iso.fit(X)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(iso, ISO_PATH)
    return rf, iso


def direct_score(feature_vec):
    """
    Direct mathematical scoring — calibrated anchor.
    Realistic score curve:
      1 breach, email-only, low sev   →  ~11-15  (LOW)
      1 breach, with password         →  ~28-35  (MEDIUM)
      3 breaches, mixed               →  ~35-42  (MEDIUM)
      5 breaches, passwords           →  ~48-55  (HIGH)
      10 breaches, high sev           →  ~63-70  (HIGH)
      20+ breaches, critical          →  ~80-90  (CRITICAL)
    """
    breach_n = feature_vec[0]
    pwd_n    = feature_vec[1]
    sev_n    = feature_vec[2]
    crit_n   = feature_vec[3]
    recent_n = feature_vec[4]
    has_pwd  = feature_vec[8]

    # Volume (0-100): 1 breach→12, 3→27, 5→32, 10→45, 20→62, 50→85
    volume = min(breach_n * 100 * 0.75, 100)

    # Password (0-100)
    password = min(pwd_n * 100, 100)

    # Severity — weighted down for single breach (avoid false HIGH)
    # sev_weight reaches 1.0 at 3+ breaches (breach_n ~ 0.27)
    sev_weight = min(0.50 + 0.50 * (breach_n / 0.27), 1.0)
    severity = sev_n * 100 * sev_weight

    # Critical bonus
    critical = min(crit_n * 100, 100)

    # Recent bonus
    recent = min(recent_n * 100, 100)

    # Weighted composite
    score = (
        volume   * 0.35 +
        password * 0.25 +
        severity * 0.22 +
        critical * 0.12 +
        recent   * 0.06
    )

    # Password breach multiplier
    if has_pwd == 1.0:
        if breach_n > 0.27:   # 3+ breaches
            score = min(score * 1.22, 100)
        else:                  # 1-2 breaches
            score = min(score * 1.10, 100)

    # Hard minimum for any real exposure
    score = max(score, 8.0)

    return min(round(float(score)), 100)


def compute_score(rf, iso, feature_vec):
    """
    Blend RF probabilities (40%) + direct score (60%) for reliable output.
    IsolationForest provides anomaly boost for unusual patterns.
    """
    X = np.array([feature_vec], dtype=np.float32)
    proba_arr = rf.predict_proba(X)[0]
    classes   = list(rf.classes_)

    p = {"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0}
    for cls, prob in zip(classes, proba_arr):
        if cls in p:
            p[cls] = prob

    rf_score = (
        p["low"]      * CENTROIDS["low"]      +
        p["medium"]   * CENTROIDS["medium"]   +
        p["high"]     * CENTROIDS["high"]     +
        p["critical"] * CENTROIDS["critical"]
    )

    d_score = direct_score(feature_vec)

    # IsolationForest anomaly boost (capped at +10)
    iso_raw   = iso.score_samples(X)[0]
    iso_boost = min(abs(iso_raw) * 12, 10) if iso_raw < -0.15 else 0.0

    # Blended: direct score is the anchor, RF refines it
    blended = rf_score * 0.40 + d_score * 0.60 + iso_boost

    final = min(max(round(float(blended)), 8), 100)
    return final, p


def compute_shap(rf, feature_vec):
    if not SHAP_AVAILABLE:
        return None
    try:
        explainer = shap.TreeExplainer(rf)
        X = np.array([feature_vec], dtype=np.float32)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            classes = list(rf.classes_)
            for target in ["critical", "high"]:
                if target in classes:
                    idx = classes.index(target)
                    shap_vals = sv[idx][0]
                    break
            else:
                shap_vals = sv[-1][0]
        else:
            shap_vals = sv[0]
        return dict(zip(FEATURE_NAMES, [float(v) * 100 for v in shap_vals]))
    except Exception as e:
        sys.stderr.write(f"[SHAP] Warning: {e}\n")
        return None


def factor_score(v):
    return min(int(round(float(v) * 10)), 10)


def build_factors(nf):
    return [
        {"icon": "🔑", "name": "PASSWORD LEAKS",  "score": factor_score(nf.get("password_norm", 0)),   "barColor": "var(--accent-red)"},
        {"icon": "💀", "name": "BREACH SEVERITY", "score": factor_score(nf.get("severity_norm", 0)),   "barColor": "var(--accent-orange)"},
        {"icon": "🔁", "name": "EXPOSURE COUNT",  "score": factor_score(nf.get("breach_norm", 0)),     "barColor": "var(--accent-yellow)"},
        {"icon": "⚡", "name": "RECENT BREACHES", "score": factor_score(nf.get("recent_norm", 0)),     "barColor": "var(--accent-cyan)"},
        {"icon": "🌐", "name": "PUBLIC EXPOSURE", "score": factor_score(nf.get("public_exposure", 0)), "barColor": "var(--accent-blue)"},
    ]


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "Empty input"})); sys.exit(1)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parse error: {e}"})); sys.exit(1)

    nf = payload.get("normalized_features", payload)
    feature_vec = [float(nf.get(name, 0.0)) for name in FEATURE_NAMES]

    # ── HARD ZERO GATE ────────────────────────────────────────────────────────
    if is_zero_vector(feature_vec):
        print(json.dumps(zero_result()))
        sys.exit(0)

    # ── Pure Python fallback (scikit-learn unavailable) ───────────────────────
    if not ML_AVAILABLE:
        score = direct_score(feature_vec)
        risk  = ("CRITICAL"  if score >= 75 else
                 "HIGH RISK" if score >= 50 else
                 "MEDIUM"    if score >= 25 else "LOW RISK")
        shap_factors = [
            {"label": f, "pts": int(abs(v * 25)), "pct": int(abs(v * 100))}
            for f, v in zip(FEATURE_NAMES, feature_vec)
        ]
        shap_factors.sort(key=lambda x: -x["pts"])
        print(json.dumps({
            "score": score, "risk_level": risk,
            "factors": build_factors(nf),
            "shap_factors": shap_factors,
        }))
        sys.exit(0)

    # ── Full ML path ──────────────────────────────────────────────────────────
    rf, iso = get_or_train_models()

    # Force retrain if old 3-class model still cached
    if hasattr(rf, 'classes_') and len(rf.classes_) != 4:
        rf, iso = get_or_train_models(force_retrain=True)

    score, proba = compute_score(rf, iso, feature_vec)

    risk = ("CRITICAL"  if score >= 75 else
            "HIGH RISK" if score >= 50 else
            "MEDIUM"    if score >= 25 else "LOW RISK")

    shap_dict = compute_shap(rf, feature_vec)
    if shap_dict:
        shap_factors = [
            {"label": k, "pts": max(int(round(abs(v))), 0), "pct": int(min(abs(v), 100))}
            for k, v in shap_dict.items()
        ]
    else:
        shap_factors = [
            {"label": f, "pts": int(abs(v * 25)), "pct": int(abs(v * 100))}
            for f, v in zip(FEATURE_NAMES, feature_vec)
        ]
    shap_factors.sort(key=lambda x: -x["pts"])

    print(json.dumps({
        "score":        score,
        "risk_level":   risk,
        "factors":      build_factors(nf),
        "shap_factors": shap_factors,
        "proba":        {k: round(v, 4) for k, v in proba.items()},
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()