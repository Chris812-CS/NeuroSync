# CSI/CVI Unknown-Group Dashboard

A Streamlit port of `csi_cvi_analysis_unknown_group_v2.ipynb`. Same parsing,
CSI/CVI/BCEA math, hypothesis flags, MAD-z pool comparison, GMM clustering,
and personalized recommendations as the notebook — just interactive, with a
session-detail view and a multi-session overview.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Using it

1. The sidebar **Data folder** field defaults to the same `DATA_DIR` path
   used in the notebook. Every `.csv` directly inside that folder is picked
   up automatically — no upload step. Change the path if you're running this
   on a different machine or pointing at a different folder, then hit
   **🔄 Reload folder** (it also auto-reloads if a file in the folder is
   added, edited, or removed).
2. **Overview** tab: pool-relative comparison table, raw values, BIC-selected
   clustering, and cluster feature profiles.
3. **Session detail** tab: pick a participant to see their CSI/CVI/HR/accuracy
   headline numbers, cluster membership with a plain-language explanation of
   *why* they landed in that cluster, ADHD/autism hypothesis-check flags, and
   the five personalized recommendations.
4. Filename-derived group/severity stays hidden everywhere (matching the
   notebook's blind design) until you flip **🔓 Reveal filename-derived
   groups** in the sidebar — that unlocks the crosscheck accuracy, cross-tab,
   flag-ratio-by-group table, and the CSI phase-shift bar chart.
5. Tunables (rolling window, gap threshold, settle time, flag tolerance, max
   clusters) live in the sidebar and match the notebook's `CONFIG` cell.

Note: since the folder path is read directly off the machine running the
app, this only works when run locally (or on a machine that can see that
path) — it can't be hosted somewhere else and point at your PC's folder.

## Files

- `app.py` — Streamlit UI
- `csi_cvi_pipeline.py` — the notebook's analysis logic, ported near-verbatim
  (no Streamlit dependency, so it's easy to unit-test or reuse elsewhere)
- `requirements.txt`

## Notes

- Recommendations and cluster leans are pilot-scale, descriptive pattern
  checks — not a diagnosis, and not validated across repeated sessions (the
  app repeats this caveat, same as the notebook).
- The crosscheck accuracy is partly circular (`cluster_group_lean` is derived
  from the same small pool it's checked against) — flagged in the UI, same
  wording as the notebook.
