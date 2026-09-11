"""
CSI/CVI unknown-group analysis pipeline.

Ported near-verbatim from csi_cvi_analysis_unknown_group_v2.ipynb so the
numbers produced by the dashboard match the notebook exactly. This module
has no Streamlit dependency; it's pure data logic, imported by app.py.
"""
import io
import itertools
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
    # Kept to the 6 metrics with the largest control-vs-adhd separation
    # (|z-gap| > ~0.7) on the current 15-session pool. At n=5/group, the
    # GMM's diagonal covariance is being estimated from too little data to
    # carry 11 dimensions -- the dropped features (CSI_resting, CVI_first30,
    # CVI_overall, adhd_flag_ratio, autism_flag_ratio) had near-zero
    # separating power for control vs. adhd and were diluting the real
    # signal from these 6. Re-evaluate the z-gaps (see z-scored feature
    # means by group) if the pool composition changes substantially.
    "accuracy", "CSI_active", "HR_overall", "BCEA_resting", "reaction_time", "CVI_active",
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
# Filename parsing (group label, kept out of any display until user reveals it)
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


def compute_pairwise_zgaps(long_df, metric_keys, group_col="group", group_order=None):
    """Z-score each metric across the whole pool (population std, ddof=0
    -- same convention as build_feature_matrix's StandardScaler), then
    compute each group's mean z-score and the absolute gap between every
    pair of groups present. This is the same "z-gap" heuristic used to
    hand-pick CLUSTER_METRIC_KEYS (see its comment), generalized to every
    candidate metric and every pairwise contrast -- not just Control vs.
    ADHD -- so feature selection doesn't silently favor separating one
    pair of groups at the expense of the others."""
    df = long_df.copy()
    present = set(df[group_col].dropna().unique())
    groups_present = [g for g in (group_order or sorted(present)) if g in present]
    pairs = list(itertools.combinations(groups_present, 2))

    rows = []
    for metric in metric_keys:
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        std = vals.std(ddof=0)
        if vals.notna().sum() < 2 or not std or pd.isna(std):
            continue
        z = (vals - vals.mean()) / std
        group_means = {g: z[df[group_col] == g].mean() for g in groups_present}
        row = {"metric": metric}
        gaps = []
        for a, b in pairs:
            if pd.isna(group_means.get(a)) or pd.isna(group_means.get(b)):
                continue
            gap = abs(group_means[a] - group_means[b])
            row[f"{a}_vs_{b}"] = round(float(gap), 3)
            gaps.append(gap)
        row["max_gap"] = round(float(max(gaps)), 3) if gaps else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("max_gap", ascending=False).reset_index(drop=True)


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
#
# Task reference (for anyone reading these functions later):
#   RESTING  -> participant watches a YouTube video, no auditory/response task.
#   PASSIVE  -> same video continues; background oddball tones play every ~700ms
#               (500Hz standard / 450Hz rare deviant, ~20% of tones).
#   ACTIVE   -> gap-overlap / response-conflict task, up to 5 min:
#               a circle cue appears (green = congruent/"low" difficulty: press
#               the button on the SAME side as the star that appears 1.5s later;
#               red = incongruent/"high" difficulty: press the OPPOSITE side).
#               Cue color is currently randomized 50/50 per trial.
#   Phase transitions (RESTING->PASSIVE at the 3-min video mark, PASSIVE->ACTIVE
#   when the video ends) each trigger a 5-second pre/post RMSSD & BPM capture.
#   Distractor zones are optional screen regions registered via
#   NeuroGazeAPI.setDistractorZones() before Start is clicked; if never called,
#   no zone data is recorded.
# ------------------------------------------------------------------

def signal_decline_point(dfs, window_trials=5, decline_frac=0.15):
    """Looks for a sustained drop in ACTIVE-phase (response-conflict task)
    accuracy, using the participant's own first-window accuracy as baseline.
    Always returns a dict (never None) so the caller can explain *why* no
    decline was found, not just that one wasn't."""
    log = dfs.get("conflict_task_log")
    if log is None or log.empty or "elapsed_sec" not in log.columns:
        return {"status": "no_task_data"}

    df = log.copy()
    df["elapsed_sec"] = pd.to_numeric(df["elapsed_sec"], errors="coerce")
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1"])
    df = df.dropna(subset=["elapsed_sec"]).sort_values("elapsed_sec").reset_index(drop=True)

    n_trials = len(df)
    if n_trials < window_trials * 2:
        return {"status": "insufficient_trials", "n_trials": n_trials, "min_required": window_trials * 2}

    df["rolling_acc"] = df["correct"].rolling(window_trials, min_periods=window_trials).mean()
    baseline_acc = df["correct"].iloc[:window_trials].mean()
    total_duration_sec = df["elapsed_sec"].iloc[-1]

    threshold = baseline_acc - decline_frac
    below = df["rolling_acc"] < threshold
    if not below.any():
        return {
            "status": "stable", "n_trials": n_trials,
            "baseline_accuracy": round(baseline_acc, 3),
            "total_duration_sec": round(total_duration_sec, 1),
        }

    first_idx = below.idxmax()
    if below.loc[first_idx:].mean() < 0.5:
        return {
            "status": "stable", "n_trials": n_trials,
            "baseline_accuracy": round(baseline_acc, 3),
            "total_duration_sec": round(total_duration_sec, 1),
        }

    return {
        "status": "declined",
        "n_trials": n_trials,
        "baseline_accuracy": round(baseline_acc, 3),
        "decline_elapsed_sec": round(df.loc[first_idx, "elapsed_sec"], 1),
        "decline_accuracy": round(df.loc[first_idx, "rolling_acc"], 3),
        "total_duration_sec": round(total_duration_sec, 1),
    }


def signal_difficulty_sensitivity(dfs):
    """Compares accuracy on green/congruent (low) vs red/incongruent (high)
    response-conflict trials, and how many of each were logged."""
    log = dfs.get("conflict_task_log")
    if log is None or log.empty or "difficulty" not in log.columns:
        return None
    df = log.copy()
    df["correct"] = df["correct"].astype(str).str.lower().isin(["true", "1"])
    by_diff = df.groupby("difficulty")["correct"].agg(["mean", "count"])
    if "low" not in by_diff.index or "high" not in by_diff.index:
        return None
    return {
        "acc_low_difficulty": round(by_diff.loc["low", "mean"], 3),
        "acc_high_difficulty": round(by_diff.loc["high", "mean"], 3),
        "n_low": int(by_diff.loc["low", "count"]),
        "n_high": int(by_diff.loc["high", "count"]),
        "gap": round(by_diff.loc["low", "mean"] - by_diff.loc["high", "mean"], 3),
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


def _phase_label(phase):
    if phase == "RESTING":
        return "the calm video"
    if phase == "PASSIVE":
        return "the video with background sounds"
    if phase == "ACTIVE":
        return "the game"
    return phase


def signal_focus_by_phase(dfs):
    """Compares on-task gaze ratio across whichever phases were logged,
    from the phase_off_task_gaze_ratio section."""
    df = dfs.get("phase_off_task_gaze_ratio")
    if df is None or df.empty or "on_task_ms" not in df.columns or "off_task_ms" not in df.columns:
        return None
    d = df.copy()
    d["on_task_ms"] = pd.to_numeric(d["on_task_ms"], errors="coerce")
    d["off_task_ms"] = pd.to_numeric(d["off_task_ms"], errors="coerce")
    d["total_ms"] = d["on_task_ms"] + d["off_task_ms"]
    d = d.dropna(subset=["total_ms"])
    d = d[d["total_ms"] > 0]
    if len(d) < 2:
        return None
    d["on_task_ratio"] = d["on_task_ms"] / d["total_ms"]
    d = d.sort_values("on_task_ratio", ascending=False)
    best, worst = d.iloc[0], d.iloc[-1]
    return {
        "best": {"phase": best["phase"], "on_task_ratio": round(best["on_task_ratio"], 3)},
        "worst": {"phase": worst["phase"], "on_task_ratio": round(worst["on_task_ratio"], 3)},
        "spread": round(best["on_task_ratio"] - worst["on_task_ratio"], 3),
    }


def signal_post_response_recovery(dfs):
    """How long it takes gaze to settle back on the game after each response
    (postresponse_gaze_recovery_ms, logged per trial in conflict_task_log).

    The raw field is noisier than it looks: it's timestamped from two clocks
    that drift against each other by a few ms, so most trials -- where gaze
    was already back on the game -- come out as small positive OR NEGATIVE
    values clustered near zero, not a clean "time since response." A negative
    value isn't a real negative duration; it just means recovery was
    effectively instant. Real recovery events, when they happen, are
    unmistakably bigger (hundreds to thousands of ms) and rare -- so this
    reports "how often was there a real pause, and how long was it" instead
    of a mean, which would blend the near-zero noise floor with a handful of
    genuine outliers into a number that describes neither."""
    log = dfs.get("conflict_task_log")
    if log is None or log.empty or "postresponse_gaze_recovery_ms" not in log.columns:
        return None
    vals = pd.to_numeric(log["postresponse_gaze_recovery_ms"], errors="coerce").dropna().clip(lower=0)
    if len(vals) < 6:
        return None

    NOTABLE_MS = 100  # below this is jitter around "already on-task"
    notable = vals[vals >= NOTABLE_MS]
    if notable.empty:
        return {"instant": True, "n_trials": int(len(vals))}

    return {
        "instant": False,
        "n_trials": int(len(vals)),
        "notable_frac": round(len(notable) / len(vals), 3),
        "notable_median_ms": round(notable.median(), 1),
    }


def signal_heart_rate_response(dfs):
    """Mean raw_bpm during RESTING vs. ACTIVE, straight from main_stream --
    deliberately just a mean-of-means, not the dashboard's full rolling-
    window CSI/CVI math, to keep this simple and easy to sanity-check."""
    main = dfs.get("main_stream")
    if main is None or main.empty or "phase" not in main.columns or "raw_bpm" not in main.columns:
        return None
    df = main.copy()
    df["raw_bpm"] = pd.to_numeric(df["raw_bpm"], errors="coerce")
    df = df[df["phase"].isin(["RESTING", "ACTIVE"]) & df["raw_bpm"].notna() & (df["raw_bpm"] > 0)]
    by_phase = df.groupby("phase")["raw_bpm"].mean()
    if "RESTING" not in by_phase.index or "ACTIVE" not in by_phase.index:
        return None
    resting_bpm, active_bpm = by_phase["RESTING"], by_phase["ACTIVE"]
    delta_bpm = active_bpm - resting_bpm
    return {
        "resting_bpm": round(resting_bpm, 1),
        "active_bpm": round(active_bpm, 1),
        "delta_bpm": round(delta_bpm, 1),
        "delta_frac": round(delta_bpm / resting_bpm, 3) if resting_bpm > 0 else None,
    }


def signal_pacing_steadiness(dfs):
    """Reaction-time consistency (coefficient of variation) across the game --
    a steadiness measure independent of accuracy. CV is a standard,
    scale-free variability statistic; the 0.35 "notably uneven" cutoff is a
    generic rule of thumb, not derived from the pilot pool (too small and
    clustered to set this kind of cutoff reliably)."""
    log = dfs.get("conflict_task_log")
    if log is None or log.empty or "rt_ms" not in log.columns:
        return None
    vals = pd.to_numeric(log["rt_ms"], errors="coerce").dropna()
    if len(vals) < 8:
        return None
    mean = vals.mean()
    sd = vals.std(ddof=0)
    return {
        "mean_rt_ms": round(mean, 1),
        "sd_rt_ms": round(sd, 1),
        "cv": round(sd / mean, 3) if mean > 0 else None,
        "n_trials": int(len(vals)),
    }


# --- recommendation text (parent-facing) ----------------------------------
#
# These are written for a parent/caregiver reading a take-home report, not for
# a technical audience: plain wording, no jargon (no phase names, ms/RMSSD
# values, or condition labels), an empathetic framing of what was observed,
# and a short list of concrete, low-effort things a parent can try. Every
# sentence still traces back to a real signal, and nothing here is invented,
# but the *why* stays in code comments rather than the parent-facing text.
# A brief reminder that this is one session's snapshot, not a diagnosis, is
# included where it matters most (the difficulty and transition sections).

def recommend_session_length(sig):
    if sig["status"] in ("no_task_data", "insufficient_trials"):
        return ("How long to keep activities going: we didn't get quite enough of the game completed this time "
                "to see a clear pattern in your child's focus over time. That's completely fine; it just means "
                "we don't have a full picture yet. If your child is willing, letting them play a little longer "
                "next time will help us see when they naturally start needing a break.")

    if sig["status"] == "stable":
        return ("How long to keep activities going: your child kept a steady pace through the whole activity "
                "today, without a clear dip in performance. That's a good sign, and it suggests the current "
                "length of the activity suits them well right now, so there's no need to shorten it or add "
                "extra breaks.")

    minutes = sig["decline_elapsed_sec"] / 60
    break_after = max(5, round(minutes))
    return (
        f"How long to keep activities going: your child did well for about the first {minutes:.0f} minutes, "
        f"and then their performance dropped off and didn't fully bounce back for the rest of the session. This "
        f"is very normal and doesn't mean they weren't trying; it usually just means they'd reached their limit "
        f"for sustained focus that day. A simple thing to try: after roughly {break_after} minutes of focused "
        f"activity, offer a short 1-2 minute break (a stretch, a sip of water, or a few minutes of something "
        f"relaxed like watching a video) before asking them to focus again. Building in breaks like this can "
        f"take pressure off both of you, since a tired brain isn't a sign of not trying hard enough."
    )


def recommend_difficulty(sig):
    if sig is None:
        return ("Practicing the trickier version of the game: we didn't get enough rounds of both the easy and "
                "the trickier version of the game this time to compare them. No action needed; we'll get a "
                "clearer picture next session.")

    low_pct, high_pct = sig["acc_low_difficulty"], sig["acc_high_difficulty"]
    if sig["gap"] < 0.1:
        return ("Practicing the trickier version of the game: your child did about the same on both the easy "
                "and the trickier rounds of the game. That's a great sign: they're managing the harder rule "
                "just as well as the simple one, so there's nothing you need to change about how you play it "
                "together right now.")

    note = ""
    if high_pct < 0.15:
        note = (
            " If the trickier rounds are still very hard to get right, it may simply mean the 'do the opposite' "
            "rule hasn't quite clicked yet, rather than your child not being capable of it. A quick reminder of "
            "the rule right before playing can make a real difference."
        )
    return (
        f"Practicing the trickier version of the game: your child did well on the easy rounds ({low_pct:.0%} "
        f"correct), but found the trickier rounds, where they have to do the opposite of their first instinct, "
        f"much harder ({high_pct:.0%} correct). This is a very common pattern, especially while a child is still "
        f"building the skill of pausing before reacting. It's not a sign of a problem: it's a skill that takes "
        f"practice, the same way learning to catch a ball takes practice.\n\n"
        f"A few things that can help at home:\n"
        f"- Play mostly the easy version for a while so your child feels confident, then mix in just a few "
        f"trickier rounds at a time rather than jumping straight to a 50/50 mix.\n"
        f"- Try quick, playful practice outside of the formal session. Games like 'Simon Says, but do the "
        f"opposite' or 'if I point up, you point down' build the same pause-and-think skill in a low-pressure way.\n"
        f"- Praise the attempt to pause and think, not just getting it right. A slower, careful wrong answer is "
        f"still real progress.\n"
        f"- Keep it short and upbeat. A couple of minutes of practice a day tends to work better than one long, "
        f"frustrating session.{note}\n\n"
        f"This is one session's snapshot, not a diagnosis. If this pattern keeps showing up over several "
        f"sessions, it's worth mentioning to your child's pediatrician or a developmental specialist, who can "
        f"look at it alongside everything else they know about your child."
    )


def recommend_modality(sig):
    if sig is None:
        return ("How your child takes in instructions: we didn't get enough information this session to tell "
                "whether your child responds better to things they see or things they hear. No action needed; "
                "we'll take another look next time.")

    vis = sig.get("visual_latency_ms")
    aud = sig.get("auditory_gaze_change_frac")

    if vis is None and aud is None:
        return ("How your child takes in instructions: we didn't get enough information this session to tell "
                "whether your child responds better to things they see or things they hear.")

    weak_sound_note = ""
    if aud is not None and aud < 0.05:
        weak_sound_note = (
            " The background sounds during today's session were quite soft, so this doesn't necessarily mean "
            "sounds don't get your child's attention. It just means today's sounds may have been easy to tune out."
        )

    return (
        "How your child takes in instructions: we looked at how quickly your child's attention shifted toward "
        "something they could see versus something they could hear in the background. So far, we don't see a "
        "clear preference for one over the other. That means at this stage, either speaking to your child "
        "directly or showing them something (a picture, a gesture, pointing) should work about equally well for "
        f"getting their attention.{weak_sound_note} If you notice at home that your child reacts much faster to "
        "one or the other, for example they respond quicker when you show them something than when you call "
        "their name, it's worth leaning into whichever one seems to reach them best, especially for important "
        "reminders or instructions."
    )


def recommend_transitions(sig):
    if sig is None:
        return ("Switching between activities: we didn't catch a clear activity change this session to see how "
                "your child responds to switches. No action needed.")

    is_to_active = "ACTIVE" in sig["transition"]
    switch_context = (
        "switching from watching the video to starting the game"
        if is_to_active else
        "switching from a calm start into watching the video"
    )
    return (
        f"Switching between activities: the biggest reaction we noticed all session was right when your child "
        f"was {switch_context}. Their body showed a noticeable startle-type response at that moment. This is "
        f"common and doesn't mean anything is wrong; sudden changes can feel a little jarring for many children, "
        f"especially when they're deep in focus on something else.\n\n"
        f"A few simple things that tend to help:\n"
        f"- Give a heads-up before switching, instead of switching all at once. For example: 'in a couple of "
        f"minutes we're going to stop this and do something else.'\n"
        f"- Use a visual or verbal countdown they can follow, like counting down from 5.\n"
        f"- A short calming moment between activities (one deep breath, a stretch, or a familiar phrase you "
        f"always use) can help them feel ready before the next thing starts.\n\n"
        f"Giving advance warning like this can make transitions feel less overwhelming for your child, and it "
        f"may also mean fewer tears, refusals, or meltdowns around activity changes at home, which can take a "
        f"real load off you as well, not just your child."
    )


def recommend_distraction(sig):
    if sig is None:
        return ("Their surroundings during focused activities: we didn't track any specific distractions in "
                "your child's surroundings this session. If there's something in their everyday environment (a "
                "phone, a window, a sibling nearby) that you suspect pulls their attention away, let us know "
                "and we can check for that specifically next time.\n\n"
                "In the meantime, one thing worth trying regardless: keep phones, screens, and windows or "
                "doorways out of your child's direct line of sight during homework or focus time. It's a "
                "low-effort default that tends to help most children, since moving light and passing activity "
                "naturally pull young eyes away from what they're supposed to be doing.")

    return (
        f"Their surroundings during focused activities: during this session, your child's attention was pulled "
        f"toward one particular spot in their surroundings about {sig['fraction_of_zone_time']:.0%} of the time "
        f"when they were supposed to be focused on the activity. That tells us something in that area is likely "
        f"competing for their attention.\n\n"
        f"A few things worth trying:\n"
        f"- See if you can remove or cover whatever is in that spot during homework or focus time. For example, "
        f"put a phone in another room, close a door, or turn a screen away.\n"
        f"- Where possible, set up a simple, calm space for focused activities, without too much visual clutter "
        f"nearby. Even small changes can make a real difference for a child who's easily pulled away.\n"
        f"- You don't need to fix everything at once. Removing even one distraction at a time is a reasonable "
        f"place to start, and it can take some of the guesswork off your plate."
    )


def recommend_focus_by_phase(sig):
    if sig is None:
        return ("Where their attention held best: we didn't get enough of the session across different activities "
                "to compare focus between them this time. No action needed.")

    best_pct, worst_pct = sig["best"]["on_task_ratio"], sig["worst"]["on_task_ratio"]
    best_label, worst_label = _phase_label(sig["best"]["phase"]), _phase_label(sig["worst"]["phase"])
    if sig["spread"] < 0.1:
        return (
            f"Where their attention held best: your child's eyes stayed on-task about equally well across "
            f"everything we tried today: {best_pct:.0%} during {best_label} and {worst_pct:.0%} during "
            f"{worst_label}. That's a good sign that focus isn't tied to one particular kind of activity. "
            f"Since attention doesn't dip for any one type of task, there's no need to warm up with an easier "
            f"activity first: you can lead with whichever one matters most that day and expect it to go about "
            f"as well as anything else."
        )

    return (
        f"Where their attention held best: your child's eyes stayed on-task {best_pct:.0%} of the time during "
        f"{best_label}, compared with {worst_pct:.0%} during {worst_label}. That's a fairly normal difference: "
        f"some activities naturally hold attention better than others. If you're "
        f"choosing what to lead with during homework or practice time, starting with something closer to "
        f"{best_label} in style may help ease them in before moving to trickier or less engaging tasks."
    )


def recommend_recovery(sig):
    if sig is None:
        return ("Bouncing back after each turn: we didn't get enough completed rounds with a clear refocus moment "
                "to see a pattern here this time. No action needed.")

    if sig["instant"]:
        return ("Bouncing back after each turn: your child's eyes were essentially already back on the game right "
                "after responding, round after round, with no real lag. That's a good sign of staying engaged with "
                "the game itself between turns.")

    pct = sig["notable_frac"]
    seconds = sig["notable_median_ms"] / 1000
    if sig["notable_frac"] < 0.2:
        return (
            f"Bouncing back after each turn: most of the time, your child's eyes were already back on the game "
            f"right after responding. A handful of rounds (about {pct:.0%}) took a bit longer to refocus "
            f"(typically around {seconds:.1f} seconds), which is a completely normal, occasional dip in attention "
            f"during a repetitive game, not something to be concerned about. No changes needed here, but on the "
            f"rounds where it does take your child a moment to refocus, a short verbal cue right after they answer "
            f"('nice, now look here') can help them settle back in a little faster."
        )

    return (
        f"Bouncing back after each turn: in about {pct:.0%} of rounds, it took your child a noticeable moment "
        f"(typically around {seconds:.1f} seconds) to look back at the game after responding. Pauses like this are a "
        f"normal part of how attention naturally drifts and resets during a repetitive game. If it's helpful, "
        f"keeping rounds short and spaced out, with a brief pause between them, may make it easier for them to "
        f"stay with the game between turns."
    )


def recommend_heart_rate_response(sig):
    if sig is None:
        return ("Their body's response to the game: we didn't get clear heart-rate readings during both the calm "
                "video and the game to compare this time. No action needed.")

    if sig["delta_frac"] is None or abs(sig["delta_frac"]) < 0.05:
        return (
            f"Their body's response to the game: your child's heart rate stayed fairly steady between the calm "
            f"video (about {sig['resting_bpm']:.0f} bpm) and the game (about {sig['active_bpm']:.0f} bpm). "
            f"That's a normal, relaxed response: the game didn't seem to key them up much either way. No action "
            f"needed here; it's worth treating this as your child's calm-and-engaged baseline, so a noticeably "
            f"bigger jump in a future session would be the thing worth mentioning alongside everything else."
        )

    direction = "rose" if sig["delta_bpm"] > 0 else "dropped"
    return (
        f"Their body's response to the game: your child's heart rate {direction} from about "
        f"{sig['resting_bpm']:.0f} bpm during the calm video to about {sig['active_bpm']:.0f} bpm during the "
        f"game. A change like this is a completely normal sign of engagement or excitement: bodies naturally "
        f"rev up a little for something active or attention-demanding, similar to what happens during play or "
        f"exercise. It's not something to be concerned about on its own; it's just useful context alongside the "
        f"other patterns in this report."
    )


def recommend_pacing_steadiness(sig):
    if sig is None:
        return ("How steady their pace was: we didn't get enough timed responses in the game to look at pacing "
                "this time. No action needed.")

    seconds = sig["mean_rt_ms"] / 1000
    if sig["cv"] is None or sig["cv"] < 0.35:
        return (
            f"How steady their pace was: your child responded at a fairly steady pace all game, averaging about "
            f"{seconds:.2f} seconds per round without a lot of swinging between very fast and very slow responses. A "
            f"steady rhythm like this is a good sign of settled, sustained attention. No changes needed here; if "
            f"this steady pace holds up across future sessions, it's a good sign the current activity length and "
            f"format suit your child well right now."
        )

    return (
        f"How steady their pace was: your child's response times varied quite a bit round to round "
        f"(averaging about {seconds:.2f} seconds), but with some rounds much faster or slower than others. This kind of "
        f"up-and-down pacing is common and doesn't mean anything is wrong; it can simply mean attention drifted in "
        f"and out a little during the game, which is normal for a repetitive task. If it's helpful, keeping rounds "
        f"short and spaced out (rather than one long stretch) may help even out the pace."
    )
