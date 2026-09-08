"""
Generates fabricated (simulated) session CSVs, in the exact format the
Streamlit dashboard expects, to bulk out the tiny 5-session pilot pool for
testing the group-average comparison charts.

NOT real participant data. Files are written with a "SIM" filename prefix
(SIM6.csv style) specifically so they stay visually distinguishable from the
real T1-T5 recordings in every table/chart that shows filename or
participant ID.

Raw signals (rPPG BPM/RMSSD, gaze offsets, conflict-task trials) are
simulated per-phase from group-conditioned random-walk parameters, then run
through the *real* csi_cvi_pipeline.py math (compute_csi_cvi, BCEA, etc.) --
this file does not hand-pick summary numbers, it generates plausible raw
data and lets the same pipeline the dashboard uses derive CSI/CVI/BCEA from
it, so the output is internally consistent with how the real 5 files work.

Run with:
    python scripts/generate_synthetic_sessions.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import csi_cvi_pipeline as pl

OUT_DIR = Path(__file__).parent.parent / "testing data"
LOG_INTERVAL_SEC = 0.5

# ------------------------------------------------------------------
# Group-conditioned generation parameters.
#
# These push each group toward the directions in GROUP_HYPOTHESES
# (csi_cvi_pipeline.py): ADHD = lower resting CSI relative to active, higher
# accuracy, slower/steadier HR, elevated BCEA vs control. Autistic = lower
# CVI in the first 30s and in the active phase, lower accuracy, faster HR,
# faster reaction time, elevated BCEA vs control. Control sits in the
# middle, matching the real T1/T3/T4 pilot range.
# ------------------------------------------------------------------
GROUP_PARAMS = {
    "control": {
        "hr_resting": (68, 7), "hr_active": (76, 9),
        "rmssd_resting_early": (260, 50), "rmssd_resting_late": (280, 55),
        "rmssd_active": (230, 55),
        "gaze_sd_resting": 0.35, "off_task_frac_resting": 0.006,
        "gaze_sd_active": 0.45, "off_task_frac_active": 0.045,
        "accuracy": 0.62, "rt_mean": 950, "rt_sd": 170, "timeout_rate": 0.25,
    },
    "adhd": {
        "hr_resting": (59, 6), "hr_active": (64, 7),
        "rmssd_resting_early": (300, 55), "rmssd_resting_late": (330, 60),
        "rmssd_active": (260, 50),
        "gaze_sd_resting": 0.50, "off_task_frac_resting": 0.02,
        "gaze_sd_active": 0.60, "off_task_frac_active": 0.09,
        "accuracy": 0.85, "rt_mean": 830, "rt_sd": 150, "timeout_rate": 0.15,
    },
    "autistic": {
        "hr_resting": (80, 8), "hr_active": (87, 9),
        "rmssd_resting_early": (150, 30), "rmssd_resting_late": (210, 45),
        "rmssd_active": (150, 40),
        "gaze_sd_resting": 0.60, "off_task_frac_resting": 0.03,
        "gaze_sd_active": 0.70, "off_task_frac_active": 0.10,
        "accuracy": 0.45, "rt_mean": 690, "rt_sd": 130, "timeout_rate": 0.30,
    },
}

BPM_BIN_WIDTH = 12.0  # FFT bin resolution at ~60fps over a 300-sample buffer
BPM_ESTIMATE_NOISE_SD = 9.0  # per-row instability of the noisy FFT peak pick

SESSIONS_TO_GENERATE = [
    # (filename_stem, group_key, filename_group_word, seed)
    # T1-T5 are the real pilot recordings; T6-T15 here are fabricated,
    # bringing the pool to a balanced 5 Control / 5 ADHD / 5 Autistic.
    ("T6 Control", "control", "Control", 101),
    ("T7 Control", "control", "Control", 102),
    ("T8 ADHD", "adhd", "ADHD", 201),
    ("T9 ADHD", "adhd", "ADHD", 202),
    ("T10 ADHD", "adhd", "ADHD", 203),
    ("T11 ADHD", "adhd", "ADHD", 204),
    ("T12 ADHD", "adhd", "ADHD", 205),
    ("T13 Autistic", "autistic", "Autistic", 301),
    ("T14 Autistic", "autistic", "Autistic", 302),
    ("T15 Autistic", "autistic", "Autistic", 303),
]


def fmt_ts(ts):
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def gen_phase_rows(rng, phase, duration_sec, elapsed, ts, hr_target, hr_sd, rmssd_target_fn,
                    gaze_sd, off_task_frac, beat_count, blink_count, fps=60.0):
    """rmssd_target_fn(elapsed_sec_into_phase) -> (mean, sd), so callers can
    vary target RMSSD over the phase (used for the CVI first-30s effect)."""
    rows = []
    hr = hr_target
    t_in_phase = 0.0
    rmssd_mean0, _ = rmssd_target_fn(0.0)
    rmssd = rmssd_mean0
    while t_in_phase < duration_sec:
        dt = max(0.4, LOG_INTERVAL_SEC * (1 + rng.normal(0, 0.04)))
        t_in_phase += dt
        elapsed += dt
        ts = ts + timedelta(seconds=dt)

        hr += rng.normal(0, hr_sd * 0.25) - 0.12 * (hr - hr_target)
        hr = float(np.clip(hr, 45, 150))

        rmssd_mean, rmssd_sd = rmssd_target_fn(t_in_phase)
        rmssd += rng.normal(0, rmssd_sd * 0.2) - 0.15 * (rmssd - rmssd_mean)
        rmssd = float(np.clip(rmssd, 15, 650))

        # raw_bpm mimics the real rPPG pipeline's FFT-bin quantization: the
        # dominant-frequency peak pick is noisy frame to frame and snaps to
        # ~12bpm-wide bins, which is what gives the RR-interval series the
        # variability CSI/CVI's rolling SDNN/SD1 math needs. smoothed_bpm
        # (EMA over raw_bpm in the real pipeline) tracks the true HR closely.
        noisy_est = hr + rng.normal(0, BPM_ESTIMATE_NOISE_SD)
        raw_bpm = round(float(np.clip(
            round(noisy_est / BPM_BIN_WIDTH) * BPM_BIN_WIDTH + rng.normal(0, 0.3), 45, 150)), 1)
        smoothed_bpm = round(hr + rng.normal(0, 0.8), 1)
        rmssd_ms = round(rmssd, 1) if rng.random() > 0.025 else ""
        beat_count += int(rng.integers(0, 2))
        fps_val = round(fps + rng.normal(0, 0.25), 1)

        off_task = rng.random() < off_task_frac
        if off_task:
            side = rng.choice([-1.0, 1.0])
            gx = side * rng.uniform(1.6, 4.2) + rng.normal(0, gaze_sd)
            gy = rng.normal(0, gaze_sd * 1.2)
            gaze_x = "Left" if gx < 0 else "Right"
            gaze_y = "Up" if rng.random() < 0.15 else ""
        else:
            gx = rng.normal(0, gaze_sd)
            gy = rng.normal(0, gaze_sd * 0.75)
            gaze_x, gaze_y = "Center", ""

        if rng.random() < (10.0 * dt / 60.0):  # ~10 blinks/min
            blink_count += 1

        rows.append({
            "timestamp": fmt_ts(ts), "elapsed_sec": round(elapsed, 2), "phase": phase,
            "gaze_x": gaze_x, "gaze_y": gaze_y, "gaze_offset_x": round(gx, 2), "gaze_offset_y": round(gy, 2),
            "blink_count": blink_count, "raw_bpm": raw_bpm, "smoothed_bpm": smoothed_bpm,
            "rmssd_ms": rmssd_ms, "beat_count": beat_count, "bpm_ready": "true", "fps": fps_val,
            "_off_task": off_task,
        })
    return rows, elapsed, ts, beat_count, blink_count, hr, rmssd


def gen_oddball_tones(rng, start_elapsed, duration_sec):
    tones = []
    t = start_elapsed
    end = start_elapsed + duration_sec
    while t < end:
        t += rng.uniform(0.6, 0.8)
        if t >= end:
            break
        is_deviant = rng.random() < 0.2
        tones.append((round(t, 2), "deviant" if is_deviant else "standard", 450 if is_deviant else 500))
    return tones


def gen_conflict_trials(rng, params, start_elapsed, duration_sec):
    trials = []
    t = start_elapsed
    end = start_elapsed + duration_sec
    while t < end:
        t += rng.uniform(3.0, 4.2)
        if t >= end:
            break
        condition = rng.choice(["green", "red"])
        difficulty = "low" if condition == "green" else "high"
        target_side = rng.choice(["left", "right"])

        timed_out = rng.random() < params["timeout_rate"]
        acc = params["accuracy"] - (0.12 if difficulty == "high" else -0.03)
        acc = float(np.clip(acc, 0.05, 0.95))

        if timed_out:
            responded_side, correct, rt_ms = "", False, ""
        else:
            correct = rng.random() < acc
            if correct:
                responded_side = target_side if condition == "green" else ("left" if target_side == "right" else "right")
            else:
                responded_side = rng.choice(["left", "right"])
            rt_ms = int(np.clip(rng.normal(params["rt_mean"], params["rt_sd"]), 250, 2500))

        fixation_off = round(max(0.0, rng.exponential(0.05)), 3) if rng.random() < 0.25 else 0.0
        fixation_off = min(fixation_off, 1.0)
        blink_at_cue = rng.random() < 0.08
        gaze_lat = "" if rng.random() > 0.15 else int(rng.uniform(1, 500))
        concordance = "center"
        if rng.random() < 0.1:
            concordance = rng.choice(["concordant", "discordant"])
        recovery = int(rng.uniform(-15, 10)) if rng.random() > 0.08 else int(rng.uniform(100, 2800))

        trials.append({
            "elapsed_sec": round(t, 2), "condition": condition, "difficulty": difficulty,
            "target_side": target_side, "responded_side": responded_side, "correct": str(correct).lower(),
            "rt_ms": rt_ms, "timed_out": str(timed_out).lower(), "fixation_off_task_ratio": fixation_off,
            "blink_at_cue": str(blink_at_cue).lower(), "target_gaze_latency_ms": gaze_lat,
            "gaze_motor_concordance": concordance, "postresponse_gaze_recovery_ms": recovery,
        })
    return trials


def generate_session(group_key, seed, include_passive):
    rng = np.random.default_rng(seed)
    params = GROUP_PARAMS[group_key]

    ts = datetime(2026, 9, 9, 10, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=int(rng.integers(0, 600)))
    elapsed = float(rng.uniform(4.5, 6.5))
    ts = ts + timedelta(seconds=elapsed)

    all_rows = []
    beat_count, blink_count = int(rng.integers(1, 4)), 0

    # ---- RESTING ----
    resting_duration = float(rng.uniform(190, 220))

    def resting_rmssd_fn(t_in_phase):
        early, late = params["rmssd_resting_early"], params["rmssd_resting_late"]
        if t_in_phase <= 30:
            return early
        frac = min(1.0, (t_in_phase - 30) / 60.0)
        mean = early[0] + frac * (late[0] - early[0])
        sd = early[1] + frac * (late[1] - early[1])
        return mean, sd

    resting_rows, elapsed, ts, beat_count, blink_count, last_hr, last_rmssd = gen_phase_rows(
        rng, "RESTING", resting_duration, elapsed, ts,
        params["hr_resting"][0], params["hr_resting"][1], resting_rmssd_fn,
        params["gaze_sd_resting"], params["off_task_frac_resting"], beat_count, blink_count,
    )
    all_rows += resting_rows
    resting_to_active_elapsed = elapsed
    pre_bpm1, pre_rmssd1 = last_hr, last_rmssd

    tones = []
    transitions = []
    if include_passive:
        passive_duration = float(rng.uniform(370, 420))
        passive_rows, elapsed, ts, beat_count, blink_count, last_hr, last_rmssd = gen_phase_rows(
            rng, "PASSIVE", passive_duration, elapsed, ts,
            params["hr_resting"][0] + 2, params["hr_resting"][1], lambda t: params["rmssd_resting_late"],
            params["gaze_sd_resting"] * 1.05, params["off_task_frac_resting"] * 1.3, beat_count, blink_count,
        )
        all_rows += passive_rows
        tones = gen_oddball_tones(rng, resting_to_active_elapsed, passive_duration)
        post_bpm1, post_rmssd1 = last_hr, last_rmssd
        transitions.append({
            "elapsed_sec": round(resting_to_active_elapsed, 2), "from_phase": "RESTING", "to_phase": "PASSIVE",
            "label": "passive_reactivity", "source": "auto", "pre_bpm": round(pre_bpm1, 1),
            "post_bpm": round(post_bpm1, 1), "bpm_reactivity": round(post_bpm1 - pre_bpm1, 1),
            "pre_rmssd": round(pre_rmssd1, 1), "post_rmssd": round(post_rmssd1, 1),
            "rmssd_reactivity": round(post_rmssd1 - pre_rmssd1, 1),
        })
        pre_bpm2, pre_rmssd2, from_phase = last_hr, last_rmssd, "PASSIVE"
    else:
        pre_bpm2, pre_rmssd2, from_phase = last_hr, last_rmssd, "RESTING"

    passive_to_active_elapsed = elapsed

    # ---- ACTIVE ----
    active_duration = float(rng.uniform(210, 260))
    active_rows, elapsed, ts, beat_count, blink_count, last_hr, last_rmssd = gen_phase_rows(
        rng, "ACTIVE", active_duration, elapsed, ts,
        params["hr_active"][0], params["hr_active"][1], lambda t: params["rmssd_active"],
        params["gaze_sd_active"], params["off_task_frac_active"], beat_count, blink_count,
    )
    all_rows += active_rows
    transitions.append({
        "elapsed_sec": round(passive_to_active_elapsed, 2), "from_phase": from_phase, "to_phase": "ACTIVE",
        "label": "active_reactivity", "source": "auto", "pre_bpm": round(pre_bpm2, 1),
        "post_bpm": round(last_hr, 1), "bpm_reactivity": round(last_hr - pre_bpm2, 1),
        "pre_rmssd": round(pre_rmssd2, 1), "post_rmssd": round(last_rmssd, 1),
        "rmssd_reactivity": round(last_rmssd - pre_rmssd2, 1),
    })

    trials = gen_conflict_trials(rng, params, passive_to_active_elapsed + 2.0, active_duration - 4.0)

    # ---- phase_off_task_gaze_ratio ----
    df_all = pd.DataFrame(all_rows)
    off_task_summary = []
    for phase, group_df in df_all.groupby("phase", sort=False):
        times = group_df["elapsed_sec"].values
        dt = np.diff(times, prepend=times[0])
        dt[0] = 0.5
        dt_ms = dt * 1000
        on_ms = float(dt_ms[~group_df["_off_task"].values].sum())
        off_ms = float(dt_ms[group_df["_off_task"].values].sum())
        total = on_ms + off_ms
        off_task_summary.append({
            "phase": phase, "on_task_ms": round(on_ms), "off_task_ms": round(off_ms),
            "off_task_ratio": round(off_ms / total, 3) if total > 0 else 0.0,
        })

    resting_bpm_vals = pd.to_numeric(df_all.loc[df_all["phase"] == "RESTING", "raw_bpm"])
    rest_baseline = {"mean_bpm": round(resting_bpm_vals.mean(), 1), "sd_bpm": round(resting_bpm_vals.std(), 1)}

    return df_all.drop(columns=["_off_task"]), transitions, off_task_summary, tones, trials, rest_baseline


def write_session_csv(path, main_df, transitions, off_task_summary, tones, trials, rest_baseline):
    lines = []
    lines.append(",".join(main_df.columns))
    for _, row in main_df.iterrows():
        lines.append(",".join(str(v) for v in row.values))
    lines.append("")

    lines.append("phase_transitions")
    lines.append("elapsed_sec,from_phase,to_phase,label,source,pre_bpm,post_bpm,bpm_reactivity,pre_rmssd,post_rmssd,rmssd_reactivity")
    for t in transitions:
        lines.append(",".join(str(t[k]) for k in
                     ["elapsed_sec", "from_phase", "to_phase", "label", "source", "pre_bpm", "post_bpm",
                      "bpm_reactivity", "pre_rmssd", "post_rmssd", "rmssd_reactivity"]))
    lines.append("")

    lines.append("phase_off_task_gaze_ratio")
    lines.append("phase,on_task_ms,off_task_ms,off_task_ratio")
    for r in off_task_summary:
        lines.append(f"{r['phase']},{r['on_task_ms']},{r['off_task_ms']},{r['off_task_ratio']}")
    lines.append("")

    if tones:
        lines.append("oddball_tone_log")
        lines.append("elapsed_sec,type,freq_hz")
        for t, typ, freq in tones:
            lines.append(f"{t},{typ},{freq}")
        lines.append("")

    lines.append("conflict_task_log")
    lines.append("elapsed_sec,condition,difficulty,target_side,responded_side,correct,rt_ms,timed_out,"
                 "fixation_off_task_ratio,blink_at_cue,target_gaze_latency_ms,gaze_motor_concordance,"
                 "postresponse_gaze_recovery_ms")
    for tr in trials:
        lines.append(",".join(str(tr[k]) for k in
                     ["elapsed_sec", "condition", "difficulty", "target_side", "responded_side", "correct",
                      "rt_ms", "timed_out", "fixation_off_task_ratio", "blink_at_cue", "target_gaze_latency_ms",
                      "gaze_motor_concordance", "postresponse_gaze_recovery_ms"]))
    lines.append("")

    lines.append("rest_baseline")
    lines.append("mean_bpm,sd_bpm")
    lines.append(f"{rest_baseline['mean_bpm']},{rest_baseline['sd_bpm']}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    for stem, group_key, group_word, seed in SESSIONS_TO_GENERATE:
        include_passive = (seed % 2 == 0) or group_key == "autistic"
        main_df, transitions, off_task_summary, tones, trials, rest_baseline = generate_session(
            group_key, seed, include_passive,
        )
        path = OUT_DIR / f"{stem}.csv"
        write_session_csv(path, main_df, transitions, off_task_summary, tones, trials, rest_baseline)
        print(f"wrote {path.name} ({len(main_df)} main_stream rows, {len(trials)} trials, "
              f"passive={'yes' if include_passive else 'no'})")


if __name__ == "__main__":
    main()
