"""
CSI/CVI Unknown-Group Dashboard
--------------------------------
Streamlit port of csi_cvi_analysis_unknown_group_v2.ipynb.

Run with:
    streamlit run app.py

Points at a local folder of session CSVs (same as the notebook's DATA_DIR)
and loads every .csv inside it directly -- no upload step. Group/severity is
parsed from each filename but stays hidden in every table and chart until
you open the "Reveal groups" section, mirroring the notebook's blind-analysis
structure.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import csi_cvi_pipeline as pl

DEFAULT_DATA_DIR = str(Path(__file__).parent / "testing data")

# ------------------------------------------------------------------
# Palette -- reused from the notebook's own NeuroSync chart styling
# ------------------------------------------------------------------
NAVY, TEAL, CORAL, PALE, SLATE, CREAM = (
    "#0B3D5C", "#1C7293", "#FF6F59", "#CFE8EF", "#8FAEBB", "#F7FAFC",
)

# NeuroSync brand palette -- blue / green / amber / red, each as
# (mid-tone for accents & icons, light bg for pills, dark text-on-pill)
BLUE = ("#378ADD", "#E6F1FB", "#0C447C")
GREEN = ("#639922", "#EAF3DE", "#27500A")
AMBER = ("#EF9F27", "#FAEEDA", "#633806")
RED = ("#E24B4A", "#FCEBEB", "#791F1F")

st.set_page_config(page_title="CSI/CVI Unknown-Group Dashboard", layout="wide", page_icon="🧠")

# ------------------------------------------------------------------
# CSS -- targets the current Streamlit wrapper testids explicitly and
# with !important. Newer Streamlit versions render stAppViewContainer /
# stMain / stMainBlockContainer / stHeader as separate layers on top of
# .stApp, each with their own background, so styling .stApp alone gets
# painted over and the page silently stays light/white.
# ------------------------------------------------------------------
st.markdown(f"""
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottomBlockContainer"] {{
    background-color: {CREAM} !important;
}}
[data-testid="stHeader"] {{
    background-color: transparent !important;
}}
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {{
    background-color: {NAVY} !important;
}}
[data-testid="stSidebar"] * {{ color: {PALE} !important; }}
h1, h2, h3 {{ color: {NAVY}; }}
[data-testid="stDataFrame"] div[role="columnheader"] {{
    color: white !important;
}}
[data-testid="stDataFrame"] div[role="columnheader"] * {{
    color: white !important;
}}
.metric-card {{
    background: white; border-radius: 10px; padding: 14px 18px;
    border-left: 5px solid {TEAL}; box-shadow: 0 1px 3px rgba(11,61,92,0.08);
}}
.flag-yes {{ color: #1C8C4A; font-weight: 600; }}
.flag-no {{ color: {SLATE}; }}
.flag-na {{ color: {SLATE}; font-style: italic; }}
.caution-box {{
    background: #FFF4E5; border-left: 5px solid {CORAL}; padding: 10px 16px;
    border-radius: 6px; font-size: 0.92rem;
}}
.ns-logo {{ display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }}
.ns-logo-icon {{
    width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
    background: linear-gradient(135deg, {BLUE[0]}, {GREEN[0]});
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}}
.ns-logo-text {{ font-size: 20px; font-weight: 700; line-height: 1.15; }}
.ns-logo-text .n1 {{ color: {BLUE[0]}; }}
.ns-logo-text .n2 {{ color: {GREEN[0]}; }}
.ns-logo-text .n3 {{ color: {AMBER[0]}; }}
.ns-logo-text .n4 {{ color: {RED[0]}; }}
.ns-tagline {{ font-size: 0.85rem; color: {PALE}; margin: 6px 0 2px; line-height: 1.4; }}
.ns-subcaption {{ font-size: 0.8rem; color: {SLATE}; margin-bottom: 18px; }}
.pool-pill {{ padding: 2px 8px; border-radius: 4px; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Sidebar -- branding + tunables
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'''<div class="ns-logo">
              <div class="ns-logo-icon">🧠</div>
              <div class="ns-logo-text">
                <span class="n1">Neuro</span><span class="n2">S</span><span class="n3">y</span><span class="n4">n</span><span class="n1">c</span>
              </div>
            </div>
            <p class="ns-tagline">Beyond gaze: a multimodal paradigm for early childhood ADHD and autism screening</p>
            <p class="ns-subcaption">Unknown-group pipeline -- blind until you reveal it.</p>''',
        unsafe_allow_html=True,
    )

    data_dir_str = DEFAULT_DATA_DIR

    st.markdown("### Tunables")
    with st.expander("Signal-processing window", expanded=False):
        window_sec = st.slider("CSI/CVI rolling window (s)", 10.0, 60.0, pl.CSI_CVI_WINDOW_SEC, 5.0)
        gap_threshold = st.slider("Logging-gap threshold (s)", 1.0, 15.0, pl.GAP_THRESHOLD_SEC, 1.0)
        settle_sec = st.slider("Post-gap settle time (s)", 5.0, 30.0, pl.SETTLE_SEC, 5.0)
    with st.expander("Comparison / clustering", expanded=False):
        rel_tol = st.slider("Within-session flag tolerance", 0.01, 0.10, pl.REL_TOL, 0.01)
        max_clusters = st.slider("Max clusters tried (BIC)", 2, 6, pl.MAX_CLUSTERS, 1)

    reveal = st.toggle("🔓 Reveal filename-derived groups", value=True,
                        help="Mirrors the notebook's final reveal section -- on by default.")

tunables = {"CSI_CVI_WINDOW_SEC": window_sec, "GAP_THRESHOLD_SEC": gap_threshold, "SETTLE_SEC": settle_sec}


# ------------------------------------------------------------------
# Load + process every session (cached on folder contents + tunables)
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and processing session CSVs...")
def process_sessions(file_fingerprints, tunables, rel_tol):
    """file_fingerprints: tuple of (path_str, mtime, size) -- part of the
    cache key, so editing/adding/removing a CSV in the folder invalidates
    the cache automatically."""
    session_results = {}
    parse_warnings = {}
    dfs_by_session = {}
    for path_str, _mtime, _size in file_fingerprints:
        path = Path(path_str)
        stem = path.stem
        group, severity, participant_id = pl.parse_filename(stem)
        dfs, warnings = pl.parse_sections(path)
        if warnings:
            parse_warnings[participant_id] = warnings
        try:
            metrics, rows = pl.extract_session_metrics(dfs, tunables)
        except Exception as e:
            parse_warnings.setdefault(participant_id, []).append(f"metric extraction failed: {e}")
            continue
        flag_ratios = pl.flag_match_ratios(metrics, rel_tol)
        session_results[participant_id] = {
            "metrics": metrics,
            "flag_ratios": flag_ratios,
            "group": group,
            "severity": severity,
            "filename": path.name,
        }
        dfs_by_session[participant_id] = dfs
    return session_results, dfs_by_session, parse_warnings


def zscore_band_colors(z):
    """Map a MAD-z value to (bg, text) using the four brand colors.
    Red = notable outlier either direction, amber = above pool, blue = below
    pool, green = close to the pool median."""
    if pd.isna(z):
        return None
    if abs(z) >= 1.0:
        return RED[1], RED[2]
    if z <= -0.25:
        return BLUE[1], BLUE[2]
    if z >= 0.25:
        return AMBER[1], AMBER[2]
    return GREEN[1], GREEN[2]


def style_pool_table(annotated_df, z_df):
    def styler(_):
        styles = pd.DataFrame("", index=annotated_df.index, columns=annotated_df.columns)
        for r in annotated_df.index:
            for c in annotated_df.columns:
                colors = zscore_band_colors(z_df.loc[r, c]) if r in z_df.index and c in z_df.columns else None
                styles.loc[r, c] = (
                    f"background-color:{colors[0]};color:{colors[1]};font-weight:600;"
                    if colors else ""
                )
        return styles
    return annotated_df.style.apply(styler, axis=None)


GROUP_DISPLAY_NAMES = {"control": "Control", "adhd": "ADHD", "autistic": "Autistic", "autistic_adhd": "Autistic+ADHD"}
GROUP_ORDER = ["control", "adhd", "autistic", "autistic_adhd"]
_jitter_rng = np.random.default_rng(42)


def render_group_comparison_chart(df, metrics, metric_labels, colors, ylabel, title, pct=False):
    """Grouped bar chart of per-group means (+/- SD) for one or more metrics,
    with individual session values overlaid as dots -- with only a handful of
    sessions per group, a bar-and-error-bar alone would imply more
    statistical confidence than the pool actually supports."""
    groups_present = [g for g in GROUP_ORDER if g in df["group"].unique()]
    if not groups_present:
        return None

    n_groups, n_metrics = len(groups_present), len(metrics)
    x = np.arange(n_groups)
    width = 0.7 / n_metrics

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=NAVY)
    ax.set_facecolor(NAVY)

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        offset = (i - (n_metrics - 1) / 2) * width
        bar_x = x + offset
        means, sds, per_group_vals = [], [], []
        for g in groups_present:
            vals = df.loc[df["group"] == g, metric].dropna()
            vals = vals * 100 if pct else vals
            per_group_vals.append(vals)
            means.append(vals.mean() if len(vals) else np.nan)
            sds.append(vals.std() if len(vals) > 1 else 0)
        ax.bar(bar_x, means, width, yerr=sds, label=label, color=color, zorder=3,
               capsize=4, error_kw={"ecolor": PALE, "elinewidth": 1})
        for gi, vals in enumerate(per_group_vals):
            if len(vals) == 0:
                continue
            jitter = (_jitter_rng.random(len(vals)) - 0.5) * width * 0.5
            ax.scatter(np.full(len(vals), bar_x[gi]) + jitter, vals, color="white",
                       edgecolor=NAVY, s=22, zorder=4, linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{GROUP_DISPLAY_NAMES.get(g, g)}\n(n={df.loc[df['group'] == g, 'participant'].nunique()})"
         for g in groups_present],
        fontsize=10, color="white",
    )
    ax.set_ylabel(ylabel, fontsize=11, color=PALE)
    ax.set_title(title, fontsize=12, color="white", fontweight="bold", pad=14)
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=PALE, labelsize=9)
    ax.yaxis.grid(True, color=SLATE, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    if n_metrics > 1:
        ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=PALE)
    plt.tight_layout()
    return fig


if not data_dir_str:
    st.title("CSI/CVI Unknown-Group Dashboard")
    st.info("Enter a data folder path in the sidebar to get started.")
    st.stop()

data_dir = Path(data_dir_str)
if not data_dir.exists() or not data_dir.is_dir():
    st.title("CSI/CVI Unknown-Group Dashboard")
    st.error(f"Folder not found: `{data_dir_str}`. Make sure the `testing data/` folder "
             f"sits next to `app.py` in the repo.")
    st.stop()

csv_paths = sorted(data_dir.glob("*.csv"))
if not csv_paths:
    st.title("CSI/CVI Unknown-Group Dashboard")
    st.warning(f"No .csv files found directly inside `{data_dir_str}`.")
    st.stop()

file_fingerprints = tuple(
    (str(p), p.stat().st_mtime, p.stat().st_size) for p in csv_paths
)
session_results, dfs_by_session, parse_warnings = process_sessions(file_fingerprints, tunables, rel_tol)
st.sidebar.caption(f"Loaded {len(csv_paths)} file(s) from `{data_dir_str}`")

if not session_results:
    st.error("None of the uploaded files could be parsed as session CSVs.")
    st.stop()

for pid, warns in parse_warnings.items():
    for w in warns:
        st.sidebar.warning(f"{pid}: {w}")

session_long = pl.build_session_long(session_results)
annotated_table, z_scores = pl.pool_annotated_table(session_long)
X_ai, n_missing_ai = pl.build_feature_matrix(session_long)
gmm, cluster_probs, prob_cols, n_clusters, bic_scores = pl.run_clustering(X_ai, max_clusters)
cluster_profiles = {col: pl.describe_cluster_profile(gmm, X_ai, i) for i, col in enumerate(prob_cols)}
group_severity_df, group_for_crosscheck, crosscheck, cluster_group_lean = pl.reveal_groups(session_results, cluster_probs)

st.title("CSI/CVI Unknown-Group Dashboard")
st.caption(f"{len(session_results)} session(s) loaded · window={window_sec:.0f}s · "
           f"clusters found: {n_clusters}")

tab_overview, tab_session = st.tabs(["📊 Overview", "🧾 Session detail"])


# ------------------------------------------------------------------
# OVERVIEW TAB
# ------------------------------------------------------------------
with tab_overview:
    n_by_group = group_severity_df["group"].value_counts()
    cols = st.columns(4)
    for i, (label, key, accent) in enumerate([
        ("Sessions", None, BLUE[0]), ("Control", "control", GREEN[0]),
        ("ADHD", "adhd", AMBER[0]), ("Autistic", "autistic", RED[0]),
    ]):
        with cols[i]:
            val = len(session_results) if key is None else int(n_by_group.get(key, 0))
            shown = val if reveal or key is None else "—"
            st.markdown(f'<div class="metric-card" style="border-left-color:{accent}">'
                        f'<div style="font-size:0.8rem;color:{SLATE}">{label}</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:{NAVY}">{shown}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("### Pool-relative comparison")
    st.caption("Value (percentile within this pool, MAD-z). With a pilot this small, treat as descriptive "
               "positioning, not population norms. Colors mark distance from the pool median: "
               "green = near median, blue = below, amber = above, red = notable outlier either way.")
    st.dataframe(style_pool_table(annotated_table, z_scores), use_container_width=True)

    with st.expander("Raw metric values"):
        st.dataframe(pl.raw_value_table(session_long), use_container_width=True)

    st.markdown("### Unsupervised clustering (blind to filename labels)")
    st.caption(f"The model groups the {len(session_results)} sessions by feature similarity alone -- it never "
               f"sees filename labels. It settled on **k={n_clusters} clusters** (lowest BIC over "
               f"k=2..{max_clusters}).")

    # ---- one card per cluster: n members, group lean (if revealed), top features ----
    GROUP_LEAN_COLORS = {"control": GREEN, "adhd": AMBER, "autistic": RED}
    cards_per_row = min(len(prob_cols), 4)
    for row_start in range(0, len(prob_cols), cards_per_row):
        row_cols = st.columns(cards_per_row)
        for i, col_name in enumerate(prob_cols[row_start:row_start + cards_per_row]):
            n_members = int((cluster_probs["assigned_cluster"] == col_name).sum())
            lean = cluster_group_lean.get(col_name)
            with row_cols[i]:
                if reveal and lean in GROUP_LEAN_COLORS:
                    accent = GROUP_LEAN_COLORS[lean][0]
                    match_n = int(crosscheck.loc[lean, col_name])
                    header_html = (
                        f'<div style="font-size:1.05rem;font-weight:700;color:{NAVY}">'
                        f'{GROUP_DISPLAY_NAMES.get(lean, lean)}</div>'
                        f'<div style="font-size:0.75rem;color:{SLATE}">{match_n}/{n_members} labeled {lean}</div>'
                    )
                else:
                    accent = SLATE
                    header_html = f'<div style="font-size:1.0rem;font-weight:700;color:{NAVY}">Unlabeled</div>'
                st.markdown(
                    f'<div class="metric-card" style="border-left-color:{accent};min-height:172px;margin-bottom:10px">'
                    f'<div style="font-size:0.72rem;color:{SLATE};text-transform:uppercase;letter-spacing:0.03em">'
                    f'{col_name} &middot; n={n_members}</div>'
                    f'{header_html}'
                    f'<div style="font-size:0.8rem;color:{SLATE};margin-top:8px;line-height:1.4">'
                    f'{cluster_profiles[col_name]}</div></div>',
                    unsafe_allow_html=True,
                )

    predicted_groups = cluster_probs["assigned_cluster"].map(cluster_group_lean)
    actual_groups = pd.Series({sid: res["group"] for sid, res in session_results.items()})
    match_results = pd.Series(
        [pl.match_result(actual_groups[sid], predicted_groups[sid]) for sid in cluster_probs.index],
        index=cluster_probs.index,
    )
    n_correct = int((match_results == True).sum())
    n_partial = int((match_results == "partial").sum())
    n_total = len(match_results)

    if reveal:
        st.markdown("#### 🔓 Group reveal & crosscheck")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card" style="border-left-color:{GREEN[0]}">'
                        f'<div style="font-size:0.8rem;color:{SLATE}">Exact matches</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:{NAVY}">{n_correct}/{n_total}</div></div>',
                        unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card" style="border-left-color:{AMBER[0]}">'
                        f'<div style="font-size:0.8rem;color:{SLATE}">Partial (comorbid)</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:{NAVY}">{n_partial}/{n_total}</div></div>',
                        unsafe_allow_html=True)
        with m3:
            n_leaned = len(set(cluster_group_lean.values()) & set(GROUP_LEAN_COLORS))
            st.markdown(f'<div class="metric-card" style="border-left-color:{RED[0]}">'
                        f'<div style="font-size:0.8rem;color:{SLATE}">Distinct group leans</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:{NAVY}">{n_leaned}/{len(prob_cols)} clusters</div></div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="caution-box">cluster→group lean was derived by majority vote over this '
                    'same small pool, so matching predictions against it is partly circular -- a pilot '
                    'sanity check, not independent validation.</div>', unsafe_allow_html=True)

        with st.expander("Cross-tab: cluster vs. filename group"):
            st.dataframe(crosscheck, use_container_width=True)

    with st.expander("Session-by-session cluster assignment & full probabilities"):
        table = cluster_probs.copy()
        table.insert(0, "confidence", cluster_probs[prob_cols].max(axis=1))
        if reveal:
            table.insert(0, "actual_group", actual_groups)
            table.insert(2, "predicted_group", predicted_groups)
            table.insert(3, "match", match_results)
        st.caption("`confidence` is the model's estimated probability for the assigned cluster; the "
                   "cluster_N columns are the full per-cluster probabilities (rows sum to 1).")
        st.dataframe(table.style.format({"confidence": "{:.0%}"}), use_container_width=True)

    with st.expander("How was k chosen? (BIC)"):
        if bic_scores:
            bic_df = pd.DataFrame({"k": list(bic_scores.keys()), "BIC": list(bic_scores.values())})
            st.bar_chart(bic_df.set_index("k"), color=TEAL)
        st.caption("Lower BIC = better fit, penalized for model complexity. k is picked automatically as "
                   f"the lowest-BIC option over k=2..{max_clusters}.")

    if reveal:
        st.markdown("#### Hidden flag-ratios vs. revealed group")
        flag_ratio_by_group = session_long[["participant", "adhd_flag_ratio", "autism_flag_ratio"]].copy()
        flag_ratio_by_group["group"] = flag_ratio_by_group["participant"].map(group_for_crosscheck)
        flag_summary = flag_ratio_by_group.groupby("group")[["adhd_flag_ratio", "autism_flag_ratio"]].mean().round(3)
        flag_summary["n"] = flag_ratio_by_group.groupby("group").size()
        st.dataframe(flag_summary, use_container_width=True)

        st.markdown("### Group averages: Control vs. Autistic")
        st.caption("Mean +/- SD per group, with each session plotted as a dot. With only a handful of sessions "
                   "per group, treat these as descriptive comparisons for this pilot pool, not statistically "
                   "validated group differences.")
        session_long_grouped = session_long.copy()
        session_long_grouped["group"] = session_long_grouped["participant"].map(group_for_crosscheck)

        gc1, gc2 = st.columns(2)
        with gc1:
            fig = render_group_comparison_chart(
                session_long_grouped, ["CSI_resting", "CSI_active"], ["Resting", "Active"], [TEAL, CORAL],
                "CSI (Cardiac Sympathetic Index)", "CSI: resting vs. active, by group",
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
        with gc2:
            fig = render_group_comparison_chart(
                session_long_grouped, ["CVI_first30", "CVI_overall"], ["First 30s", "Overall"], [BLUE[0], GREEN[0]],
                "CVI (Cardiac Vagal Index)", "CVI: first 30s vs. overall, by group",
            )
            if fig:
                st.pyplot(fig, use_container_width=True)

        gc3, gc4 = st.columns(2)
        with gc3:
            fig = render_group_comparison_chart(
                session_long_grouped, ["HR_overall"], ["HR overall"], [AMBER[0]],
                "Heart rate (bpm)", "Heart rate, by group",
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
        with gc4:
            fig = render_group_comparison_chart(
                session_long_grouped, ["BCEA_resting"], ["BCEA resting"], [RED[0]],
                "BCEA (gaze instability)", "Gaze instability (resting), by group",
            )
            if fig:
                st.pyplot(fig, use_container_width=True)

        gc5, gc6 = st.columns(2)
        with gc5:
            fig = render_group_comparison_chart(
                session_long_grouped, ["accuracy"], ["Accuracy"], [GREEN[0]],
                "Accuracy (%)", "Task accuracy, by group", pct=True,
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
        with gc6:
            fig = render_group_comparison_chart(
                session_long_grouped, ["reaction_time"], ["Reaction time"], [BLUE[0]],
                "Reaction time (ms)", "Reaction time, by group",
            )
            if fig:
                st.pyplot(fig, use_container_width=True)

        fig = render_group_comparison_chart(
            session_long_grouped, ["adhd_flag_ratio", "autism_flag_ratio"], ["ADHD flag ratio", "Autism flag ratio"],
            [AMBER[0], RED[0]], "Flag ratio (%)", "Hidden hypothesis flag ratios, by group", pct=True,
        )
        if fig:
            st.pyplot(fig, use_container_width=True)

        st.markdown("#### CSI phase shift by session")
        chart_sessions = list(session_results.keys())
        labels = [f"{sid}\n({session_results[sid]['group']})" for sid in chart_sessions]
        resting = [session_results[sid]["metrics"].get("CSI_resting") for sid in chart_sessions]
        active = [session_results[sid]["metrics"].get("CSI_active") for sid in chart_sessions]

        fig, ax = plt.subplots(figsize=(9, 5.0), facecolor=NAVY)
        ax.set_facecolor(NAVY)
        x = np.arange(len(chart_sessions))
        width = 0.32
        b1 = ax.bar(x - width / 2, resting, width, label="Resting phase", color=TEAL, zorder=3)
        b2 = ax.bar(x + width / 2, active, width, label="Active phase", color=CORAL, zorder=3)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if np.isnan(h):
                    continue
                ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", fontsize=11, color=PALE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11, color="white")
        ax.set_ylabel("CSI (Cardiac Sympathetic Index)", fontsize=11, color=PALE)
        ax.set_title("Within-Session Phase Shift in CSI\n(each participant scored against their own resting baseline)",
                     fontsize=13, color="white", fontweight="bold", pad=16)
        ax.spines[:].set_visible(False)
        ax.tick_params(colors=PALE, labelsize=10)
        ax.yaxis.grid(True, color=SLATE, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=PALE)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Flip **Reveal filename-derived groups** in the sidebar to see group labels, cluster crosscheck "
                "accuracy, and the CSI phase-shift chart.")


# ------------------------------------------------------------------
# SESSION DETAIL TAB
# ------------------------------------------------------------------
with tab_session:
    sess_id = st.selectbox("Session", list(session_results.keys()))
    res = session_results[sess_id]
    metrics = res["metrics"]
    dfs = dfs_by_session[sess_id]

    header = f"### {sess_id}"
    if reveal:
        header += f"  ·  group: **{res['group']}**" + (f" ({res['severity']})" if res['severity'] else "")
    st.markdown(header)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CSI (resting → active)",
              f"{metrics.get('CSI_resting', float('nan')):.2f} → {metrics.get('CSI_active', float('nan')):.2f}")
    m2.metric("CVI overall", f"{metrics.get('CVI_overall', float('nan')):.2f}")
    m3.metric("HR overall (bpm)", f"{metrics.get('HR_overall', float('nan')):.1f}")
    m4.metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.0%}" if pd.notna(metrics.get("accuracy")) else "n/a")

    st.markdown("#### Cluster membership")
    st.caption("Unsupervised -- fit blind to filename labels.")
    if sess_id in cluster_probs.index:
        probs_row = cluster_probs.loc[sess_id]
        for col in prob_cols:
            lean = f" (leans '{cluster_group_lean[col]}')" if reveal else ""
            st.progress(float(probs_row[col]), text=f"{col}{lean} -- {probs_row[col]:.0%}  ·  {cluster_profiles[col]}")
        assigned = probs_row["assigned_cluster"]
        st.markdown(f"**Assigned cluster:** `{assigned}` -- {cluster_profiles[assigned]}"
                    + (f"  →  predicted group **{cluster_group_lean[assigned]}**" if reveal else ""))

        explanation = pl.explain_cluster_assignment(gmm, X_ai, cluster_probs, sess_id)
        if explanation is not None:
            with st.expander(f"Why {explanation['assigned_col']} over {explanation['runner_up_col']}?"):
                st.markdown("**In favor of the assigned cluster:**")
                st.dataframe(explanation["for_assigned"][["feature", "value_z", "dist_to_assigned", "dist_to_runner_up"]],
                             use_container_width=True, hide_index=True)
                if len(explanation["against_assigned"]):
                    st.markdown("**Pulled the other way (outweighed):**")
                    st.dataframe(explanation["against_assigned"][["feature", "value_z", "dist_to_assigned", "dist_to_runner_up"]],
                                 use_container_width=True, hide_index=True)

    st.markdown("#### Hypothesis checks")
    hc1, hc2 = st.columns(2)
    for col, group_name in [(hc1, "adhd"), (hc2, "autism")]:
        with col:
            st.markdown(f"**{group_name.upper()} pattern checks**")
            flags = pl.flag_hypothesis_directions(metrics, group_name, rel_tol)
            for _, row in flags.iterrows():
                if row["flag"] == "YES":
                    st.markdown(f'<span class="flag-yes">✓ YES</span> -- {row["hypothesis"]}', unsafe_allow_html=True)
                elif row["flag"] == "no":
                    st.markdown(f'<span class="flag-no">✗ no</span> -- {row["hypothesis"]}', unsafe_allow_html=True)
                elif str(row["flag"]).startswith("value="):
                    st.markdown(f'<span class="flag-na">{row["flag"]}</span> -- {row["hypothesis"]} '
                                f'<span style="color:{SLATE};font-size:0.85rem">(needs comparison session)</span>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="flag-na">n/a</span> -- {row["hypothesis"]}', unsafe_allow_html=True)

    def render_recommendation(n, signal_fn, recommend_fn):
        """Compute + render one recommendation; a bad signal/metric for this
        particular session shouldn't take down the rest of the report, and
        Streamlit Cloud redacts real exception text from the UI, so surface
        it here directly to make production issues diagnosable."""
        try:
            text = recommend_fn(signal_fn())
        except Exception as e:
            text = f"_Couldn't generate this recommendation ({type(e).__name__}: {e})._"
        st.markdown(f"{n}. " + text)

    st.markdown("#### Personalized recommendations")
    render_recommendation(1, lambda: pl.signal_decline_point(dfs), pl.recommend_session_length)
    render_recommendation(2, lambda: pl.signal_difficulty_sensitivity(dfs), pl.recommend_difficulty)
    render_recommendation(3, lambda: pl.signal_modality(metrics, dfs), pl.recommend_modality)
    render_recommendation(4, lambda: pl.signal_transition_reactivity(dfs), pl.recommend_transitions)
    render_recommendation(5, lambda: pl.signal_distraction(dfs), pl.recommend_distraction)
    render_recommendation(6, lambda: pl.signal_focus_by_phase(dfs), pl.recommend_focus_by_phase)
    render_recommendation(7, lambda: pl.signal_post_response_recovery(dfs), pl.recommend_recovery)
    render_recommendation(8, lambda: pl.signal_heart_rate_response(dfs), pl.recommend_heart_rate_response)
    render_recommendation(9, lambda: pl.signal_pacing_steadiness(dfs), pl.recommend_pacing_steadiness)

    st.markdown(
        f'<div class="caution-box">Confidence: LOW -- based on a single session, from a pool of '
        f'{len(session_results)} session(s) total. Cluster membership and recommendations are descriptive, '
        f'pattern-based suggestions for this sitting only -- not a diagnosis, and not validated across '
        f'repeated sessions.</div>', unsafe_allow_html=True,
    )

    with st.expander("Raw session tables (main_stream + logged sections)"):
        for name, df in dfs.items():
            st.markdown(f"**{name}** ({len(df)} rows)")
            st.dataframe(df.head(50), use_container_width=True)
