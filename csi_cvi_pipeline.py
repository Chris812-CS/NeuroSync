"""
CSI/CVI unknown-group analysis pipeline.

Ported near-verbatim from csi_cvi_analysis_unknown_group_v2.ipynb so the
numbers produced by the dashboard match the notebook exactly. This module
has no Streamlit dependency -- it's pure data logic, imported by app.py.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ------------------------------------------------------------------
# Tunables (defaults match the notebook; app.py can override via sidebar)
# ------------------------------------------------------------------
CSI_CVI_WINDOW_SEC = 30.0
GAP_THRESHOLD_SEC = 5.0
SETTLE_SEC = 15.0
REL_TOL = 0.03
MAD_SCALE = 1.4826
MAX_CLUSTERS = 4

SECTION_NAMES = {
    "phase_transitions",
    "phase_off_task_gaze_ratio",
    "oddball_tone_log",
    "conflict_task_log",
    "distractor_zone_fixation",
    "trial_log",
    "rest_baseline",
}

AUTISM_KEYWORDS = ["autistic", "autism", "asd"]
ADHD_KEYWORDS = ["adhd"]
SEVERITY_KEYWORDS = ["mild", "moderate", "severe"]
CONTROL_KEYWORDS = ["control"]

METRIC_KEYS = [
    "CSI_resting", "CSI_passive", "CSI_active",
    "CVI_first30", "CVI_rest_of_session", "CVI_passive", "CVI_active", "CVI_overall",
    "HR_resting", "HR_overall",
    "accuracy", "reaction_time",
    "gaze_speed_pre_tone", "gaze_speed_post_tone",
    "BCEA_resting", "BCEA_overall",
]

CLUSTER_METRIC_KEYS = [
    "CSI_resting", "CSI_active",
    "CVI_first30", "CVI_active", "CVI_overall",
    "HR_overall", "accuracy", "reaction_time", "BCEA_resting",
    "adhd_flag_ratio", "autism_flag_ratio",
]

GROUP_HYPOTHESES = {
    "adhd": [
        {"metric": "CSI_resting_vs_active", "desc": "lower CSI in resting phase", "comparison": "within_session"},
        {"metric": "CSI_passive_vs_active", "desc": "reduced CSI in passive phase", "comparison": "within_session"},
        {"metric": "accuracy", "desc": "higher accuracy", "comparison": "value_only"},
        {"metric": "heart_rate", "desc": "slower heart rate (slowest HR)", "comparison": "value_only"},
        {"metric": "sound_gaze_effect", "desc": "eye movement speed stayed the same", "comparison": "within_session"},
        {"metric": "BCEA_resting", "desc": "elevated gaze instability (BCEA) vs control", "comparison": "value_only"},
    ],
    "autism": [
        {"metric": "CVI_first30_vs_rest", "desc": "lower CVI in first 30s before task starts", "comparison": "within_session"},
        {"metric": "CVI_active_vs_passive", "desc": "reduced CVI in active phase", "comparison": "within_session"},
        {"metric": "CVI_overall", "desc": "lower CVI the entire time", "comparison": "value_only"},
        {"metric": "accuracy", "desc": "lower accuracy", "comparison": "value_only"},
        {"metric": "heart_rate", "desc": "faster heart rate", "comparison": "value_only"},
        {"metric": "reaction_time", "desc": "faster reaction time", "comparison": "value_only"},
        {"metric": "BCEA_resting", "desc": "elevated gaze instability (BCEA) vs control", "comparison": "value_only"},
    ],
}


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------
def parse_sections(file_like_or_path):
    """Split one exported CSV (path or file-like/bytes) into its component tables."""
    if isinstance(file_like_or_path, (str, Path)):
        with open(file_like_or_path) as f:
            lines = f.readlines()
    else:
        # Streamlit UploadedFile or bytes
        raw = file_like_or_path.getvalue() if hasattr(file_like_or_path, "getvalue") else file_like_or_path
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        lines = raw.splitlines(keepends=True)

    sections = {}
    current_name = "main_stream"
    current_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in SECTION_NAMES:
            sections[current_name] = current_lines
            current_name = stripped
            current_lines = []
            continue
        if stripped == "":
            continue
        current_lines.append(line)
    sections[current_name] = current_lines

    dfs = {}
    warnings = []
    for name, lns in sections.items():
        if not lns:
            continue
        txt = "".join(lns)
        try:
            dfs[name] = pd.read_csv(io.StringIO(txt))
        except Exception as e:
            warnings.append(f"could not parse section '{name}': {e}")
    return dfs, warnings


# ------------------------------------------------------------------
# CSI / CVI
# ------------------------------------------------------------------
def compute_csi_cvi(dfs, window_sec=CSI_CVI_WINDOW_SEC, gap_threshold_sec=GAP_THRESHOLD_SEC,
                     settle_sec=SETTLE_SEC):
    main = dfs.get("main_stream")
    if main is None:
        raise ValueError("No main_stream section in this file.")

    df = main.copy()
    df["elapsed_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")
    df["raw_bpm"] = pd.to_numeric(df["raw_bpm"], errors="coerce")
    df["rmssd_ms"] = pd.to_numeric(df["rmssd_ms"], errors="coerce")
    df = df.dropna(subset=["elapsed_sec"]).sort_values("elapsed_sec").reset_index(drop=True)

    diffs = df["elapsed_sec"].diff()
    gap_end_times = df.loc[diffs > gap_threshold_sec, "elapsed_sec"].tolist()

    df["excluded_settling"] = False
    for gap_end in gap_end_times:
        df.loc[
            (df["elapsed_sec"] >= gap_end) & (df["elapsed_sec"] < gap_end + settle_sec),
            "excluded_settling",
        ] = True

    df["SD1"] = df["rmssd_ms"] / np.sqrt(2)
    df["est_RR"] = 60000 / df["raw_bpm"]
    df.loc[df["excluded_settling"], ["SD1", "est_RR"]] = np.nan

    d = df.set_index(pd.to_timedelta(df["elapsed_sec"], unit="s"))
    d["rolling_SDNN"] = d["est_RR"].rolling(f"{window_sec:.0f}s", min_periods=5).std()
    d["rolling_SD1"] = d["SD1"].rolling(f"{window_sec:.0f}s", min_periods=5).mean()

    sd2_variance = 2 * d["rolling_SDNN"] ** 2 - d["rolling_SD1"] ** 2
    d["SD2"] = np.sqrt(np.maximum(sd2_variance, 0))
    d["SD2_undefined"] = sd2_variance < 0

    d["CSI"] = d["SD2"] / d["rolling_SD1"]
    d["CVI"] = np.where(d["SD2"] > 0, np.log10(16 * d["rolling_SD1"] * d["SD2"]), np.nan)

    d = d.reset_index(drop=True)

    return {
        "gap_end_times": gap_end_times,
        "n_excluded_settling": int(df["excluded_settling"].sum()),
        "rows": d,
    }


def compute_bcea(x, y, confidence=0.68):
    df = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"),
    }).dropna()

    if len(df) < 5:
        return np.nan

    sx, sy = df["x"].std(), df["y"].std()
    rho = df["x"].corr(df["y"])
    if pd.isna(rho) or sx == 0 or sy == 0:
        return np.nan

    k = -np.log(1 - confidence)
    return float(2 * k * np.pi * sx * sy * np.sqrt(max(1 - rho ** 2, 0)))


# ------------------------------------------------------------------
# Single-session metric extraction
# ------------------------------------------------------------------
def _find_phase_label(phase_values, keywords):
    for val in phase_values:
        low = str(val).lower()
        if any(k in low for k in keywords):
            return val
    return None


def _find_column(columns, keywords):
    for col in columns:
        low = str(col).lower()
        if any(k in low for k in keywords):
            return col
    return None


def extract_session_metrics(dfs, tunables=None):
    tunables = tunables or {}
    result = compute_csi_cvi(
        dfs,
        window_sec=tunables.get("CSI_CVI_WINDOW_SEC", CSI_CVI_WINDOW_SEC),
        gap_threshold_sec=tunables.get("GAP_THRESHOLD_SEC", GAP_THRESHOLD_SEC),
        settle_sec=tunables.get("SETTLE_SEC", SETTLE_SEC),
    )
    d = result["rows"]

    phases = d["phase"].dropna().unique().tolist() if "phase" in d.columns else []
    resting_phase = _find_phase_label(phases, ["rest", "baseline"])
    passive_phase = _find_phase_label(phases, ["passive"])
    active_phase = _find_phase_label(phases, ["active", "conflict", "task"])

    def phase_mean(col, phase_label):
        if phase_label is None or col not in d.columns:
            return np.nan
        return d.loc[d["phase"] == phase_label, col].mean()

    metrics = {}
    metrics["_phases_detected"] = phases
    metrics["_resting_phase"] = resting_phase
    metrics["_passive_phase"] = passive_phase
    metrics["_active_phase"] = active_phase
    metrics["_n_excluded_settling"] = result["n_excluded_settling"]
    metrics["_gap_end_times"] = result["gap_end_times"]

    metrics["CSI_resting"] = phase_mean("CSI", resting_phase)
    metrics["CSI_passive"] = phase_mean("CSI", passive_phase)
    metrics["CSI_active"] = phase_mean("CSI", active_phase)

    metrics["CVI_first30"] = d.loc[d["elapsed_sec"] <= 30, "CVI"].mean() if "elapsed_sec" in d.columns else np.nan
    metrics["CVI_rest_of_session"] = d.loc[d["elapsed_sec"] > 30, "CVI"].mean() if "elapsed_sec" in d.columns else np.nan
    metrics["CVI_passive"] = phase_mean("CVI", passive_phase)
    metrics["CVI_active"] = phase_mean("CVI", active_phase)
    metrics["CVI_overall"] = d["CVI"].mean() if "CVI" in d.columns else np.nan

    metrics["HR_resting"] = phase_mean("raw_bpm", resting_phase)
    metrics["HR_overall"] = d["raw_bpm"].mean() if "raw_bpm" in d.columns else np.nan
    if phases and "raw_bpm" in d.columns:
        by_phase_hr = d.groupby("phase")["raw_bpm"].mean()
        metrics["HR_min_phase"] = by_phase_hr.idxmin() if not by_phase_hr.empty else None
        metrics["HR_min_value"] = by_phase_hr.min() if not by_phase_hr.empty else np.nan

    metrics["accuracy"] = np.nan
    metrics["reaction_time"] = np.nan
    for section_name in ("trial_log", "conflict_task_log"):
        sec = dfs.get(section_name)
        if sec is None or sec.empty:
            continue
        acc_col = _find_column(sec.columns, ["correct", "accuracy", "is_hit", "hit"])
        rt_col = _find_column(sec.columns, ["reaction_time", "rt_ms", "rt", "latency", "response_time"])
        if acc_col is not None and np.isnan(metrics["accuracy"]):
            vals = pd.to_numeric(sec[acc_col], errors="coerce")
            metrics["accuracy"] = vals.mean()
            metrics["accuracy_source"] = f"{section_name}.{acc_col}"
        if rt_col is not None and np.isnan(metrics["reaction_time"]):
            vals = pd.to_numeric(sec[rt_col], errors="coerce")
            metrics["reaction_time"] = vals.mean()
            metrics["reaction_time_source"] = f"{section_name}.{rt_col}"

    metrics["gaze_speed_pre_tone"] = np.nan
    metrics["gaze_speed_post_tone"] = np.nan
    tone_log = dfs.get("oddball_tone_log")

    gaze_speed_col = _find_column(d.columns, ["gaze_speed", "gaze_velocity", "eye_speed", "eye_velocity"])

    if gaze_speed_col is None:
        offset_x_col = _find_column(d.columns, ["gaze_offset_x", "offset_x"])
        offset_y_col = _find_column(d.columns, ["gaze_offset_y", "offset_y"])
        if offset_x_col is not None and offset_y_col is not None and "elapsed_sec" in d.columns:
            dx = d[offset_x_col].astype(float).diff()
            dy = d[offset_y_col].astype(float).diff()
            dt = d["elapsed_sec"].diff()
            speed = np.sqrt(dx ** 2 + dy ** 2) / dt.where(dt > 0)
            d = d.copy()
            d["_derived_gaze_speed"] = speed
            gaze_speed_col = "_derived_gaze_speed"

    if tone_log is not None and not tone_log.empty and gaze_speed_col is not None:
        tone_time_col = _find_column(tone_log.columns, ["elapsed_sec", "time"])
        if tone_time_col is not None:
            tone_times = pd.to_numeric(tone_log[tone_time_col], errors="coerce").dropna().tolist()
            WINDOW = 2.0
            pre_vals, post_vals = [], []
            for t in tone_times:
                pre = d.loc[(d["elapsed_sec"] >= t - WINDOW) & (d["elapsed_sec"] < t), gaze_speed_col]
                post = d.loc[(d["elapsed_sec"] > t) & (d["elapsed_sec"] <= t + WINDOW), gaze_speed_col]
                pre_vals.extend(pre.dropna().tolist())
                post_vals.extend(post.dropna().tolist())
            if pre_vals and post_vals:
                metrics["gaze_speed_pre_tone"] = float(np.mean(pre_vals))
                metrics["gaze_speed_post_tone"] = float(np.mean(post_vals))

    offset_x_col = _find_column(d.columns, ["gaze_offset_x", "offset_x"])
    offset_y_col = _find_column(d.columns, ["gaze_offset_y", "offset_y"])
    metrics["BCEA_resting"] = np.nan
    metrics["BCEA_passive"] = np.nan
    metrics["BCEA_active"] = np.nan
    metrics["BCEA_overall"] = np.nan

    if offset_x_col is not None and offset_y_col is not None:

        def bcea_for_phase(phase_label):
            if phase_label is None or "phase" not in d.columns:
                return np.nan
            sub = d.loc[d["phase"] == phase_label]
            return compute_bcea(sub[offset_x_col], sub[offset_y_col])

        metrics["BCEA_resting"] = bcea_for_phase(resting_phase)
        metrics["BCEA_passive"] = bcea_for_phase(passive_phase)
        metrics["BCEA_active"] = bcea_for_phase(active_phase)
        metrics["BCEA_overall"] = compute_bcea(d[offset_x_col], d[offset_y_col])

    return metrics, d


# ------------------------------------------------------------------
# Hypothesis flags
# ------------------------------------------------------------------
def _rel_lower(a, b, rel_tol=REL_TOL):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return None
    return (b - a) / abs(b) > rel_tol


def flag_hypothesis_directions(metrics, group, rel_tol=REL_TOL):
    rows = []

    if group == "adhd":
        checks = [
            ("CSI_resting_vs_active", "lower CSI in resting phase",
             _rel_lower(metrics.get("CSI_resting"), metrics.get("CSI_active"), rel_tol)),
            ("CSI_passive_vs_active", "reduced CSI in passive phase",
             _rel_lower(metrics.get("CSI_passive"), metrics.get("CSI_active"), rel_tol)),
        ]
        pre, post = metrics.get("gaze_speed_pre_tone"), metrics.get("gaze_speed_post_tone")
        if pd.notna(pre) and pd.notna(post) and pre != 0:
            pct_change = abs(post - pre) / abs(pre)
            checks.append(("sound_gaze_effect", "Eye speed stayed the same", pct_change <= rel_tol))
        else:
            checks.append(("sound_gaze_effect", "Eye speed stayed the same", None))
        value_only = [
            ("accuracy", "higher accuracy", metrics.get("accuracy")),
            ("heart_rate", "slower heart rate", metrics.get("HR_overall")),
            ("BCEA_resting", "elevated gaze instability (BCEA)", metrics.get("BCEA_resting")),
        ]

    elif group == "autism":
        checks = [
            ("CVI_first30_vs_rest", "lower CVI in first 30s",
             _rel_lower(metrics.get("CVI_first30"), metrics.get("CVI_rest_of_session"), rel_tol)),
            ("CVI_active_vs_passive", "reduced CVI in active phase",
             _rel_lower(metrics.get("CVI_active"), metrics.get("CVI_passive"), rel_tol)),
        ]
        value_only = [
            ("CVI_overall", "lower CVI the entire time", metrics.get("CVI_overall")),
            ("accuracy", "lower accuracy", metrics.get("accuracy")),
            ("heart_rate", "faster heart rate", metrics.get("HR_overall")),
            ("reaction_time", "faster reaction time", metrics.get("reaction_time")),
            ("BCEA_resting", "elevated gaze instability (BCEA)", metrics.get("BCEA_resting")),
        ]
    else:
        raise ValueError("group must be 'adhd' or 'autism'")

    for key, desc, flag in checks:
        flag_str = "n/a (insufficient data)" if flag is None else ("YES" if flag else "no")
        rows.append({"metric": key, "hypothesis": desc, "flag": flag_str, "comparison": "within_session"})

    for key, desc, val in value_only:
        val_str = "n/a" if val is None or (isinstance(val, float) and pd.isna(val)) else round(val, 3)
        rows.append({"metric": key, "hypothesis": desc, "flag": f"value={val_str}", "comparison": "needs comparison session"})

    return pd.DataFrame(rows)


def flag_match_ratios(metrics, rel_tol=REL_TOL):
    adhd_flags = flag_hypothesis_directions(metrics, "adhd", rel_tol)
    autism_flags = flag_hypothesis_directions(metrics, "autism", rel_tol)
    adhd_within = adhd_flags[adhd_flags["comparison"] == "within_session"]
    autism_within = autism_flags[autism_flags["comparison"] == "within_session"]
    adhd_yes, adhd_total = (adhd_within["flag"] == "YES").sum(), len(adhd_within)
    autism_yes, autism_total = (autism_within["flag"] == "YES").sum(), len(autism_within)
    return {
        "adhd_flag_ratio": adhd_yes / adhd_total if adhd_total else np.nan,
        "autism_flag_ratio": autism_yes / autism_total if autism_total else np.nan,
    }


# ------------------------------------------------------------------
# Filename parsing (group label -- kept out of any display until user reveals it)
# ------------------------------------------------------------------
def parse_filename(stem):
    low = stem.lower()
    has_autism = any(k in low for k in AUTISM_KEYWORDS)
    has_adhd = any(k in low for k in ADHD_KEYWORDS)

    if has_autism and has_adhd:
        group = "autistic_adhd"
    elif has_autism:
        group = "autistic"
    elif has_adhd:
        group = "adhd"
    else:
        group = "control"

    severity = next((k for k in SEVERITY_KEYWORDS if k in low), None)

    remainder = stem
    for w in AUTISM_KEYWORDS + ADHD_KEYWORDS + SEVERITY_KEYWORDS + CONTROL_KEYWORDS:
        remainder = re.sub(rf"\b{re.escape(w)}\b", "", remainder, flags=re.IGNORECASE)
    participant_id = re.sub(r"\s+", " ", remainder).strip()
    if not participant_id:
        participant_id = stem

    return group, severity, participant_id


# ------------------------------------------------------------------
# Pool comparison
# ------------------------------------------------------------------
def build_session_long(session_results, metric_keys=METRIC_KEYS):
    records = []
    for sid, res in session_results.items():
        row = {"participant": sid}
        row.update({k: res["metrics"].get(k) for k in metric_keys})
        row.update(res["flag_ratios"])
        records.append(row)
    return pd.DataFrame(records)


def raw_value_table(long_df, metric_keys=METRIC_KEYS):
    return long_df.set_index("participant")[metric_keys].T.round(3)


def mad_z(series, mad_scale=MAD_SCALE):
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(np.nan, index=series.index)
    return (series - med) / (mad * mad_scale)


def pool_annotated_table(long_df, metric_keys=METRIC_KEYS, mad_scale=MAD_SCALE):
    df = long_df.set_index("participant")[metric_keys]
    pct = df.rank(pct=True) * 100
    z = df.apply(lambda s: mad_z(s, mad_scale))

    annotated = pd.DataFrame(index=metric_keys, columns=df.index, dtype=object)
    for metric in metric_keys:
        for pid in df.index:
            v, p, zz = df.loc[pid, metric], pct.loc[pid, metric], z.loc[pid, metric]
            if pd.isna(v):
                annotated.loc[metric, pid] = "n/a"
            else:
                annotated.loc[metric, pid] = f"{v:.2f} (p{p:.0f}, z={zz:+.2f})"
    return annotated, z.T


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------
def build_feature_matrix(long_df, metric_keys=CLUSTER_METRIC_KEYS):
    df = long_df.set_index("participant")
    X_raw = df[metric_keys]
    n_missing = X_raw.isna().sum()

    scaler = StandardScaler()
    X = pd.DataFrame(
        scaler.fit_transform(X_raw.fillna(X_raw.mean())),
        index=X_raw.index, columns=metric_keys,
    )
    return X, n_missing


def select_k_by_bic(X, k_range):
    scores = {}
    for k in k_range:
        if k >= len(X):
            continue
        m = GaussianMixture(n_components=k, covariance_type="diag", random_state=0, n_init=10)
        m.fit(X.values)
        scores[k] = m.bic(X.values)
    if not scores:
        return 2, {}
    return min(scores, key=scores.get), scores


def run_clustering(X_ai, max_clusters=MAX_CLUSTERS):
    k_range = range(2, max_clusters + 1)
    if len(X_ai) > 2:
        n_clusters, bic_scores = select_k_by_bic(X_ai, k_range)
    else:
        n_clusters, bic_scores = 2, {}

    gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=0, n_init=10)
    gmm.fit(X_ai.values)

    cluster_probs = pd.DataFrame(
        gmm.predict_proba(X_ai.values),
        index=X_ai.index,
        columns=[f"cluster_{i}" for i in range(n_clusters)],
    ).round(3)
    prob_cols = [c for c in cluster_probs.columns]
    cluster_probs["assigned_cluster"] = cluster_probs[prob_cols].idxmax(axis=1)

    return gmm, cluster_probs, prob_cols, n_clusters, bic_scores


def describe_cluster_profile(gmm, X_ai, cluster_idx, top_n=3):
    means = pd.Series(gmm.means_[cluster_idx], index=X_ai.columns)
    top = means.reindex(means.abs().sort_values(ascending=False).index).head(top_n)
    parts = []
    for feat, z in top.items():
        direction = "high" if z > 0 else "low"
        parts.append(f"{direction} {feat} (z={z:+.2f})")
    return ", ".join(parts)


def reveal_groups(session_results, cluster_probs):
    group_severity_df = pd.DataFrame({
        sid: {"group": res["group"], "severity": res["severity"] or "-"}
        for sid, res in session_results.items()
    }).T
    group_for_crosscheck = group_severity_df["group"]
    crosscheck = pd.crosstab(group_for_crosscheck, cluster_probs["assigned_cluster"])

    cluster_group_lean = {}
    for cluster_col in crosscheck.columns:
        cluster_group_lean[cluster_col] = crosscheck[cluster_col].idxmax()
    prob_cols = [c for c in cluster_probs.columns if c != "assigned_cluster"]
    for col in prob_cols:
        cluster_group_lean.setdefault(col, "unknown")

    return group_severity_df, group_for_crosscheck, crosscheck, cluster_group_lean


def match_result(actual, predicted):
    if actual == predicted:
        return True
    if actual == "autistic_adhd" and predicted in ("autistic", "adhd"):
        return "partial"
    return False


def explain_cluster_assignment(gmm, X_ai, cluster_probs, sess_id, top_n=5):
    if sess_id not in cluster_probs.index:
        return None

    probs_row = cluster_probs.loc[sess_id]
    cluster_cols = [c for c in cluster_probs.columns if c != "assigned_cluster"]
    ranked = probs_row[cluster_cols].sort_values(ascending=False)
    if len(ranked) < 2:
        return None
    assigned_col, runner_up_col = ranked.index[0], ranked.index[1]
    assigned_idx = cluster_cols.index(assigned_col)
    runner_idx = cluster_cols.index(runner_up_col)

    x = X_ai.loc[sess_id]
    rows = []
    for j, feat in enumerate(X_ai.columns):
        xi = x[feat]
        mean_a, var_a = gmm.means_[assigned_idx, j], gmm.covariances_[assigned_idx, j]
        mean_r, var_r = gmm.means_[runner_idx, j], gmm.covariances_[runner_idx, j]
        ll_a = -0.5 * np.log(2 * np.pi * var_a) - (xi - mean_a) ** 2 / (2 * var_a)
        ll_r = -0.5 * np.log(2 * np.pi * var_r) - (xi - mean_r) ** 2 / (2 * var_r)
        rows.append({
            "feature": feat, "value_z": xi,
            "dist_to_assigned": abs(xi - mean_a), "dist_to_runner_up": abs(xi - mean_r),
            "loglik_diff": ll_a - ll_r,
        })

    feat_df = pd.DataFrame(rows).sort_values("loglik_diff", ascending=False)

    return {
        "assigned_col": assigned_col,
        "runner_up_col": runner_up_col,
        "for_assigned": feat_df[feat_df["loglik_diff"] > 0].head(top_n),
        "against_assigned": feat_df[feat_df["loglik_diff"] < 0].sort_values("loglik_diff").head(top_n),
    }


# ------------------------------------------------------------------
# Recommendation signals + text
# ------------------------------------------------------------------
def signal_decline_point(dfs, window_trials=5, decline_frac=0.15):
    log = dfs.get("conflict_task_log")
    if log is None or log.empty or "elapsed_sec" not in log.columns:
        return None

    df = log.copy()
    df["elapsed_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1"])
    df = df.dropna(subset=["elapsed_sec"]).sort_values("elapsed_sec").reset_index(drop=True)

    if len(df) < window_trials * 2:
        return None

    df["rolling_acc"] = df["correct"].rolling(window_trials, min_periods=window_trials).mean()
    baseline_acc = df["correct"].iloc[:window_trials].mean()

    threshold = baseline_acc - decline_frac
    below = df["rolling_acc"] < threshold
    if not below.any():
        return None

    first_idx = below.idxmax()
    if below.loc[first_idx:].mean() < 0.5:
        return None

    return {
        "baseline_accuracy": round(baseline_acc, 3),
        "decline_elapsed_sec": round(df.loc[first_idx, "elapsed_sec"], 1),
        "decline_accuracy": round(df.loc[first_idx, "rolling_acc"], 3),
    }


def signal_difficulty_sensitivity(dfs):
    log = dfs.get("conflict_task_log")
    if log is None or log.empty or "difficulty" not in log.columns:
        return None
    df = log.copy()
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1"])
    by_diff = df.groupby("difficulty")["correct"].mean()
    if "low" not in by_diff.index or "high" not in by_diff.index:
        return None
    return {
        "acc_low_difficulty": round(by_diff["low"], 3),
        "acc_high_difficulty": round(by_diff["high"], 3),
        "gap": round(by_diff["low"] - by_diff["high"], 3),
    }


def signal_modality(metrics, dfs):
    pre = metrics.get("gaze_speed_pre_tone")
    post = metrics.get("gaze_speed_post_tone")
    log = dfs.get("conflict_task_log")
    visual_latency = None
    if log is not None and not log.empty and "target_gaze_latency_ms" in log.columns:
        vals = pd.to_numeric(log["target_gaze_latency_ms"], errors="coerce").dropna()
        if len(vals):
            visual_latency = round(vals.mean(), 0)

    auditory_change = None
    if pd.notna(pre) and pd.notna(post) and pre:
        auditory_change = round(abs(post - pre) / abs(pre), 3)

    if visual_latency is None and auditory_change is None:
        return None
    return {"visual_latency_ms": visual_latency, "auditory_gaze_change_frac": auditory_change}


def signal_transition_reactivity(dfs):
    trans = dfs.get("phase_transitions")
    if trans is None or trans.empty:
        return None
    df = trans.copy()
    for col in ("rmssd_reactivity", "bpm_reactivity"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "rmssd_reactivity" not in df.columns or df["rmssd_reactivity"].dropna().empty:
        return None
    biggest = df.loc[df["rmssd_reactivity"].abs().idxmax()]
    return {
        "transition": f"{biggest.get('from_phase', '?')}->{biggest.get('to_phase', '?')}",
        "rmssd_reactivity_ms": round(biggest["rmssd_reactivity"], 1),
    }


def signal_distraction(dfs):
    zones = dfs.get("distractor_zone_fixation")
    if zones is None or zones.empty or "fraction_of_zone_time" not in zones.columns:
        return None
    df = zones.copy()
    df["fraction_of_zone_time"] = pd.to_numeric(df["fraction_of_zone_time"], errors="coerce")
    df = df.dropna(subset=["fraction_of_zone_time"])
    if df.empty:
        return None
    top = df.loc[df["fraction_of_zone_time"].idxmax()]
    return {"zone": top["zone"], "fraction_of_zone_time": round(top["fraction_of_zone_time"], 3)}


def recommend_session_length(sig):
    if sig is None:
        return "Session length: no clear decline point detected this session (either performance stayed stable, or there wasn't enough trial data)."
    return (f"Session length / break timing: performance held steady for the first "
            f"~{sig['decline_elapsed_sec'] / 60:.1f} min (accuracy ~{sig['baseline_accuracy']:.0%}), "
            f"then rolling accuracy dropped to ~{sig['decline_accuracy']:.0%}. "
            f"Recommend structuring sessions in ~{max(5, round(sig['decline_elapsed_sec'] / 60)):.0f}-minute "
            f"blocks with a short break between.")


def recommend_difficulty(sig):
    if sig is None:
        return "Task pacing: not enough difficulty-tagged trials to assess this session."
    if sig["gap"] < 0.1:
        return (f"Task pacing: accuracy was similar across low ({sig['acc_low_difficulty']:.0%}) and "
                f"high ({sig['acc_high_difficulty']:.0%}) difficulty trials -- no strong pacing "
                f"adjustment indicated this session.")
    return (f"Task pacing / difficulty: accuracy on low-difficulty trials "
            f"({sig['acc_low_difficulty']:.0%}) was notably higher than on high-difficulty trials "
            f"({sig['acc_high_difficulty']:.0%}). Recommend introducing harder task variants "
            f"gradually rather than mixing difficulty levels from the start.")


def recommend_modality(sig):
    if sig is None:
        return "Modality / stimulus format: insufficient data (no tone log or gaze-latency data) this session."
    parts = []
    if sig.get("visual_latency_ms") is not None:
        parts.append(f"visual cue reorientation averaged {sig['visual_latency_ms']:.0f}ms")
    if sig.get("auditory_gaze_change_frac") is not None:
        parts.append(f"auditory tones changed gaze speed by {sig['auditory_gaze_change_frac']:.0%}")
    detail = "; ".join(parts) if parts else "no clear signal"
    return (f"Modality / stimulus format: {detail}. No strong preference detected this session -- "
            f"insufficient difference to recommend one format over the other yet.")


def recommend_transitions(sig):
    if sig is None:
        return "Transitions: no phase-transition data recorded this session."
    return (f"Transitions: largest physiological shift was {sig['rmssd_reactivity_ms']:+.1f}ms RMSSD "
            f"around the {sig['transition']} transition. Recommend giving advance warning before "
            f"switching activities, rather than abrupt transitions.")


def recommend_distraction(sig):
    if sig is None:
        return "Environment / distraction sensitivity: no distractor zones were configured this session -- category not evaluated."
    return (f"Environment / distraction sensitivity: {sig['fraction_of_zone_time']:.0%} of zone-relevant "
            f"time was spent fixating on '{sig['zone']}'. Recommend a visually simplified workspace "
            f"with minimal peripheral clutter during focused tasks.")
