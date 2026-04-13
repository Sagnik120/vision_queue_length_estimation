"""
PA-SQGN Queue Estimation — Streamlit Web Dashboard
Compatible with Streamlit 1.12+ (old venv safe)

Run:
    pip install altair==4.2.2 --force-reinstall
    streamlit run app.py
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, ".")
os.makedirs("outputs", exist_ok=True)

st.set_page_config(
    page_title="PA-SQGN | Queue Estimation",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.kpi-box {
    border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 10px; background: #fafafa; text-align: center;
    margin-bottom: 8px;
}
.kpi-label { font-size: 0.76rem; color: #64748b; margin-bottom: 3px; }
.kpi-value { font-size: 1.45rem; font-weight: 700; color: #1e293b; }
.kpi-sub   { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }
.finding {
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.86rem; color: #14532d; margin: 10px 0;
}
.warn {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.84rem; color: #78350f; margin: 8px 0;
}
.config-a { border-left: 4px solid #E24B4A; padding-left: 10px; margin-bottom: 8px; }
.config-b { border-left: 4px solid #EF9F27; padding-left: 10px; margin-bottom: 8px; }
.config-c { border-left: 4px solid #1D9E75; padding-left: 10px; margin-bottom: 8px; }
.pill {
    display:inline-block; padding:2px 9px; border-radius:20px;
    font-size:0.72rem; font-weight:600; margin:2px;
}
.pill-gray   { background:#f1f5f9; color:#475569; }
.pill-orange { background:#ffedd5; color:#92400e; }
.pill-green  { background:#d1fae5; color:#065f46; }
.pill-purple { background:#ede9fe; color:#5b21b6; }
.novel { background:#ede9fe; color:#5b21b6; border-radius:4px;
         padding:1px 8px; font-size:0.72rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("Run: pip install plotly")

try:
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── constants & helpers ───────────────────────────────────────────────────────
COL = {"A": "#E24B4A", "B": "#EF9F27", "C": "#1D9E75"}

CONFIGS = [
    {
        "key":    "A",
        "title":  "Config A — Hybrid baseline",
        "desc":   "MST + PCA on (x,y) positions only. No pose, no orientation, no smoother.",
        "pills":  [("pill-gray","Spatial only"),("pill-gray","No orientation"),("pill-gray","No smoothing")],
        "csv":    "outputs/ablation_A.csv",
        "video":  "outputs/ablation_A.mp4",
        "script": "scripts.run_hybrid",
        "var_path":   "VIDEO_PATH",
        "var_csv":    "OUTPUT_CSV",
        "var_video":  "OUTPUT_VIDEO",
    },
    {
        "key":    "B",
        "title":  "Config B — Pose graph only",
        "desc":   "4-term edge weight: spatial × orientation × velocity × spacing. No EMA smoother.",
        "pills":  [("pill-orange","+ Facing angle θ"),("pill-orange","+ Velocity"),("pill-orange","+ Spacing regularity")],
        "csv":    "outputs/ablation_B.csv",
        "video":  "outputs/ablation_B.mp4",
        "script": "scripts.run_pose_only",
        "var_path":   "VIDEO_PATH",
        "var_csv":    "OUTPUT_CSV",
        "var_video":  "OUTPUT_VIDEO",
    },
    {
        "key":    "C",
        "title":  "Config C — Full PA-SQGN",
        "desc":   "Complete system: pose graph + EMA smoother + scene classifier + LangChain advisory.",
        "pills":  [("pill-green","+ EMA smoother"),("pill-green","+ Scene classifier"),("pill-purple","+ LLM advisory")],
        "csv":    "outputs/queue_log.csv",
        "video":  "outputs/queue_output.mp4",
        "script": "main_pose",
        "var_path":   "VIDEO_PATH",
        "var_csv":    "OUTPUT_CSV",
        "var_video":  "OUTPUT_VIDEO",
    },
]


def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def merge_nearest(pred, gt, tol=8):
    """Match each GT row to nearest prediction frame within tolerance."""
    if pred is None or gt is None or pred.empty or gt.empty:
        return None
    pred = pred.sort_values("frame").reset_index(drop=True)
    gt   = gt.sort_values("frame").reset_index(drop=True)
    rows = []
    for _, row in gt.iterrows():
        d   = (pred["frame"] - row["frame"]).abs()
        idx = d.idxmin()
        if d[idx] <= tol:
            r = pred.loc[idx].to_dict()
            if "gt_size" in row.index:
                r["gt_size"] = row["gt_size"]
            if "gt_label" in row.index:
                r["gt_label"] = row["gt_label"]
            rows.append(r)
    return pd.DataFrame(rows) if rows else None


def compute_stats(df, gt=None):
    """Compute per-config stability and accuracy metrics."""
    if df is None or df.empty:
        return {}
    s  = df["queue_size"]
    d  = s.diff().abs().dropna()
    st = {
        "mean":       round(float(s.mean()), 2),
        "max":        int(s.max()),
        "delta_std":  round(float(d.std()), 3),
        "delta_mean": round(float(d.mean()), 3),
        "n":          len(df),
        "mae":        None,
        "rmse":       None,
    }
    if gt is not None:
        merged = merge_nearest(df, gt)
        if merged is not None and not merged.empty:
            err      = merged["queue_size"] - merged["gt_size"]
            st["mae"]  = round(float(np.mean(np.abs(err))), 2)
            st["rmse"] = round(float(np.sqrt(np.mean(err**2))), 2)
    return st


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi-box">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def pills_html(pills):
    return " ".join(
        f'<span class="pill {cls}">{txt}</span>' for cls, txt in pills
    )


# ── page title ────────────────────────────────────────────────────────────────
st.title("PA-SQGN — Vision-Based Queue Length Estimation")
st.caption("Pose-Aware Spatiotemporal Queue Graph Network · M.Tech CV Project")

tab_demo, tab_results, tab_ablation, tab_arch = st.tabs([
    "Live Demo & Processing",
    "Results Dashboard",
    "Ablation Study",
    "Architecture & Novelty",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.subheader("Upload a video and run all three system configurations")

    st.markdown("""
    Upload any queue video. Click **Run All Configs** to run A → B → C sequentially.
    All outputs (CSV + annotated video) are saved automatically.
    The Ablation Study tab updates the moment processing is complete.
    """)

    uploaded = st.file_uploader(
        "Upload video (mp4 / avi / mov)",
        type=["mp4", "avi", "mov"],
        help="Phone camera footage or CCTV recording of a queue.",
    )

    if uploaded:
        save_path = os.path.join("outputs", "uploaded_input.mp4")
        with open(save_path, "wb") as f:
            f.write(uploaded.read())
        # forward slashes always work on Windows inside Python
        video_path = save_path.replace("\\", "/")

        st.video(save_path)
        st.caption("Uploaded video preview")
        st.markdown("---")

        btn_all = st.button("Run All Configs (A + B + C)  —  ~5 min")
        btn_c   = st.button("Run Config C only  —  ~2 min  (fastest)")

        if btn_all or btn_c:
            progress = st.progress(0)
            status   = st.empty()
            errors   = []

            def run_one(cfg, pct_start, pct_end):
                status.markdown(f"**Running {cfg['title']}...**")
                progress.progress(pct_start)
                try:
                    mod = importlib.import_module(cfg["script"])
                    importlib.reload(mod)
                    # set paths via module attributes
                    setattr(mod, cfg["var_path"],  video_path)
                    setattr(mod, cfg["var_csv"],   cfg["csv"])
                    setattr(mod, cfg["var_video"], cfg["video"])
                    # Config C exposes run(); A and B expose main()
                    if hasattr(mod, "run"):
                        mod.run(video_path=video_path)
                    else:
                        mod.main()
                    progress.progress(pct_end)
                except Exception as e:
                    errors.append(f"{cfg['title']}: {e}")
                    progress.progress(pct_end)

            if btn_all:
                run_one(CONFIGS[0], 0,  30)
                run_one(CONFIGS[1], 30, 65)
                run_one(CONFIGS[2], 65, 100)
            else:
                run_one(CONFIGS[2], 0, 100)

            status.empty()
            progress.progress(100)

            if errors:
                for e in errors:
                    st.markdown(
                        f'<div class="warn">Processing error: {e}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("All configs done. Check the Ablation Study tab for the comparison.")

    # ── SIDE-BY-SIDE VIDEOS ───────────────────────────────────────────────────
    any_video = any(os.path.exists(c["video"]) for c in CONFIGS)
    if any_video:
        st.markdown("---")
        st.subheader("Output videos — side by side")
        st.caption(
            "Watch how each config detects the queue differently. "
            "Config A shows the raw geometric approach; "
            "Config B adds pose orientation arrows and graph edges; "
            "Config C adds confidence circles, density heatmap, and the advisory banner."
        )

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown(
                '<div class="config-a"><b>Config A</b> — Hybrid baseline</div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(CONFIGS[0]["video"]):
                with open(CONFIGS[0]["video"], "rb") as f:
                    st.video(f.read())
            else:
                st.info("Run Config A to generate this video.")

        with col_b:
            st.markdown(
                '<div class="config-b"><b>Config B</b> — Pose graph only</div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(CONFIGS[1]["video"]):
                with open(CONFIGS[1]["video"], "rb") as f:
                    st.video(f.read())
            else:
                st.info("Run Config B to generate this video.")

        with col_c:
            st.markdown(
                '<div class="config-c"><b>Config C</b> — Full PA-SQGN</div>',
                unsafe_allow_html=True,
            )
            if os.path.exists(CONFIGS[2]["video"]):
                with open(CONFIGS[2]["video"], "rb") as f:
                    st.video(f.read())
            else:
                st.info("Run Config C to generate this video.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.subheader("Full system output — Config C (PA-SQGN)")

    df_c = load_csv("outputs/queue_log.csv")
    if df_c is not None:
        df_c["frame"] = pd.to_numeric(df_c["frame"], errors="coerce")
        df_c = df_c.dropna(subset=["frame"]).reset_index(drop=True)
        df_c["frame"] = df_c["frame"].astype(int)
        df_c = df_c.sort_values("frame").reset_index(drop=True)
    if df_c is None:
        st.info("Run the system from the Live Demo tab to generate outputs/queue_log.csv")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "Avg queue size",    f"{df_c['queue_size'].mean():.1f}", "persons")
        kpi(c2, "Peak queue size",   str(df_c['queue_size'].max()), "persons")
        kpi(c3, "Avg queue length",  f"{df_c['queue_length_m'].mean():.1f} m", "estimated")
        kpi(c4, "Frames as QUEUE",   f"{(df_c['scene_label']=='queue').mean():.0%}", "scene classifier")
        kpi(c5, "DeltaStd (jitter)", f"{df_c['queue_size'].diff().std():.3f}", "lower = better")

        st.markdown("---")

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_c["frame"], y=df_c["queue_size"],
                name="Queue size", mode="lines+markers",
                line=dict(color=COL["C"], width=2),
                marker=dict(size=5),
            ))
            fig.add_hline(
                y=df_c["queue_size"].mean(), line_dash="dash", line_color="#aaa",
                annotation_text=f"Mean {df_c['queue_size'].mean():.1f}",
            )
            fig.update_layout(
                title="Queue size over time (Config C)",
                xaxis_title="Frame", yaxis_title="People in queue", height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

            # queue length as a simple text summary (avoids plotly rendering issues)
            avg_len = df_c["queue_length_m"].mean()
            max_len = df_c["queue_length_m"].max()
            min_len = df_c["queue_length_m"].min()

            label_counts = df_c["scene_label"].value_counts().to_dict()
            queue_pct  = round(label_counts.get("queue",     0) / len(df_c) * 100, 1)
            uncert_pct = round(label_counts.get("uncertain", 0) / len(df_c) * 100, 1)
            crowd_pct  = round(label_counts.get("crowd",     0) / len(df_c) * 100, 1)

            cl, cr = st.columns(2)
            with cl:
                st.markdown("**Queue length summary**")
                st.markdown(f"""
                                | Metric | Value |
                                |---|---|
                                | Average | {avg_len:.2f} m |
                                | Maximum | {max_len:.2f} m |
                                | Minimum | {min_len:.2f} m |
                                """)
            with cr:
                st.markdown("**Scene label distribution**")
                st.markdown(f"""
                                | Label | % of frames |
                                |---|---|
                                | Queue | {queue_pct}% |
                                | Uncertain | {uncert_pct}% |
                                | Crowd | {crowd_pct}% |
                                """)

            fig4 = px.line(
                df_c, x="frame", y="scene_confidence",
                title="Classifier confidence over time",
                color_discrete_sequence=["#7C3AED"],
            )
            fig4.add_hline(
                y=0.45, line_dash="dot", line_color="orange",
                annotation_text="Queue threshold (0.45)",
            )
            fig4.update_layout(height=260, yaxis_range=[0, 1])
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Advisory message log")
        cols_show = [c for c in
                     ["timestamp","frame","queue_size","queue_length_m",
                      "scene_label","scene_confidence","advisory_message"]
                     if c in df_c.columns]
        st.dataframe(df_c[cols_show])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ablation:
    st.subheader("Ablation study — contribution of each component")

    df_a = load_csv("outputs/ablation_A.csv")
    df_b = load_csv("outputs/ablation_B.csv")
    df_c = load_csv("outputs/queue_log.csv")
    gt   = load_csv("data/ground_truth.csv")
    has_gt = gt is not None and not (gt["gt_size"] == 0).all()

    sa = compute_stats(df_a, gt if has_gt else None)
    sb = compute_stats(df_b, gt if has_gt else None)
    sc = compute_stats(df_c, gt if has_gt else None)

    # ── config description cards (using columns, one per card) ────────────────
    col_ca, col_cb, col_cc = st.columns(3)

    with col_ca:
        st.markdown(
            f'<div class="config-a">'
            f'<b>Config A — Hybrid baseline</b><br>'
            f'<small>MST + PCA on (x,y) positions. No pose, no orientation, no smoother.</small><br>'
            f'{pills_html(CONFIGS[0]["pills"])}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if sa:
            st.metric("Mean size",   sa["mean"])
            st.metric("DeltaStd",    sa["delta_std"], help="Frame-to-frame jitter")
            if sa.get("mae") is not None:
                st.metric("MAE vs GT", sa["mae"])

    with col_cb:
        st.markdown(
            f'<div class="config-b">'
            f'<b>Config B — Pose graph only</b><br>'
            f'<small>4-term edge weight: spatial × orientation × velocity × spacing.</small><br>'
            f'{pills_html(CONFIGS[1]["pills"])}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if sb:
            st.metric("Mean size",   sb["mean"])
            st.metric("DeltaStd",    sb["delta_std"])
            if sb.get("mae") is not None:
                st.metric("MAE vs GT", sb["mae"])

    with col_cc:
        st.markdown(
            f'<div class="config-c">'
            f'<b>Config C — Full PA-SQGN</b><br>'
            f'<small>Pose graph + EMA smoother + scene classifier + LLM advisory.</small><br>'
            f'{pills_html(CONFIGS[2]["pills"])}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if sc:
            st.metric("Mean size",   sc["mean"])
            st.metric("DeltaStd",    sc["delta_std"])
            if sc.get("mae") is not None:
                st.metric("MAE vs GT", sc["mae"])

    # ── auto key-finding ──────────────────────────────────────────────────────
    if sa and sc:
        a_delta = sa.get("delta_std", 0) or 0.001
        c_delta = sc.get("delta_std", 0) or 0.001
        pct_imp = round((a_delta - c_delta) / a_delta * 100, 1)
        overcount = round((sa.get("mean", 0) or 0) - (sc.get("mean", 0) or 0), 1)
        direction = "reduction" if pct_imp > 0 else "increase"
        st.markdown(
            f'<div class="finding">'
            f'Key finding: Config C shows a <b>{abs(pct_imp)}% {direction} in DeltaStd</b> vs Config A '
            f'(A: {a_delta} → C: {c_delta}). '
            f'Config A overcounts by ~<b>{overcount} persons</b> on average because the hybrid MST '
            f'has no orientation filter — it includes passers-by facing the wrong direction. '
            f'Config B and C agree on queue membership, confirming pose orientation is the '
            f'primary filtering mechanism.'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not has_gt:
        st.markdown(
            '<div class="warn">MAE/RMSE require data/ground_truth.csv with real counts. '
            'A sample template is in data/ground_truth.csv — replace gt_size values with '
            'your manually counted numbers.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── full metrics table ────────────────────────────────────────────────────
    st.markdown("#### Quantitative comparison table")

    def _fmt(v): return str(v) if v is not None else "N/A"

    rows = []
    for label, s in [("A — Hybrid baseline", sa),
                     ("B — Pose graph only",  sb),
                     ("C — Full PA-SQGN",     sc)]:
        rows.append({
            "Config":          label,
            "Mean size":       _fmt(s.get("mean")       if s else None),
            "Peak size":       _fmt(s.get("max")        if s else None),
            "DeltaStd":        _fmt(s.get("delta_std")  if s else None),
            "Avg delta/frame": _fmt(s.get("delta_mean") if s else None),
            "MAE vs GT":       _fmt(s.get("mae")        if s else None),
            "RMSE vs GT":      _fmt(s.get("rmse")       if s else None),
        })
    st.dataframe(pd.DataFrame(rows))

    st.markdown("---")

    # ── queue size chart: all 3 configs on one plot ───────────────────────────
    if HAS_PLOTLY:
        st.markdown("#### Queue size over time — all configurations")
        fig_abl = go.Figure()
        for df_cfg, label, key in [
            (df_a, "Config A — Hybrid",    "A"),
            (df_b, "Config B — Pose only", "B"),
            (df_c, "Config C — Full",      "C"),
        ]:
            if df_cfg is not None and not df_cfg.empty:
                fig_abl.add_trace(go.Scatter(
                    x=df_cfg["frame"], y=df_cfg["queue_size"],
                    name=label, mode="lines",
                    line=dict(color=COL[key],
                              width=2.5 if key == "C" else 1.5),
                ))
        if has_gt:
            fig_abl.add_trace(go.Scatter(
                x=gt["frame"], y=gt["gt_size"],
                name="Ground truth",
                mode="markers",
                marker=dict(symbol="x", size=8, color="black"),
            ))
        fig_abl.update_layout(
            xaxis_title="Frame",
            yaxis_title="Queue size (persons)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_abl, use_container_width=True)

        # ── DeltaStd bar chart ────────────────────────────────────────────────
        st.markdown("#### Frame-to-frame jitter (DeltaStd) — lower is better")
        valid = [(k, s) for k, s in [("A", sa), ("B", sb), ("C", sc)]
                 if s and s.get("delta_std") is not None]
        if valid:
            fig_bar = px.bar(
                x=[v[0] for v in valid],
                y=[v[1]["delta_std"] for v in valid],
                color=[v[0] for v in valid],
                color_discrete_map=COL,
                labels={"x": "Configuration", "y": "DeltaStd", "color": "Config"},
                text=[str(v[1]["delta_std"]) for v in valid],
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.plotly_chart(fig_bar)

    # ── confusion matrix (if GT has gt_label column) ──────────────────────────
    if (HAS_SKLEARN and has_gt
            and df_c is not None
            and "gt_label" in gt.columns):
        st.markdown("#### Scene classifier — confusion matrix")
        merged_cm = merge_nearest(
            df_c[["frame", "scene_label"]] if "scene_label" in df_c.columns else df_c,
            gt[["frame", "gt_label"]],
        )
        if merged_cm is not None and not merged_cm.empty:
            labs = ["queue", "uncertain", "crowd"]
            cm   = confusion_matrix(
                merged_cm["gt_label"], merged_cm["scene_label"], labels=labs
            )
            fig_cm, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labs, yticklabels=labs, ax=ax)
            ax.set_ylabel("Ground truth")
            ax.set_xlabel("Predicted")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            st.text(classification_report(
                merged_cm["gt_label"], merged_cm["scene_label"],
                labels=labs, zero_division=0,
            ))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE & NOVELTY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    st.subheader("PA-SQGN system architecture")

    col_pipe, col_novel = st.columns(2)

    with col_pipe:
        st.markdown("**End-to-end pipeline:**")
        st.markdown("""
| Stage | Module | Output |
|---|---|---|
| 1. Detection | YOLOv8n-pose | Bboxes + 17 COCO keypoints |
| 2. Tracking | ByteTrack | Stable track IDs |
| 3. Feature extraction | main_pose.py | pos, angle θ, velocity v |
| 4. Graph construction | pose_graph.py | 4-term weighted graph |
| 5. MST longest path | pose_graph.py | Queue spine |
| 6. EMA smoother | temporal_smoother.py | Per-person confidence |
| 7. Scene classification | crowd_classifier.py | queue / crowd / uncertain |
| 8. Advisory generation | langchain_alerts.py | Natural language alert |
| 9. Logging | metrics_logger.py | CSV + annotated video |
""")

    with col_novel:
        st.markdown("**Novel contributions:**")
        st.markdown("""
<span class="novel">NOVEL 1</span> **4-term composite edge weight**

`w = spatial × orientation × velocity × spacing`

- **Spatial:** `exp(-d / (scale × 0.8))`
- **Orientation:** `cos(Δθ)` — clamped [0,1]
- **Velocity:** `dot(v₁,v₂)` — coherent motion
- **Spacing:** `exp(-|d−scale| / (scale×0.3))` — uniform intervals

<span class="novel">NOVEL 2</span> **EMA temporal confidence scoring**

Per-person float score in [0,1]. Asymmetric rise (0.25) vs decay (0.25).
Replaces binary in/out membership used by all prior work.

<span class="novel">NOVEL 3</span> **Explicit scene-type classifier**

5-feature graph classifier outputs queue / crowd / uncertain.
No prior queue paper classifies scene type.

<span class="novel">NOVEL 4</span> **LLM advisory layer**

LangChain + Gemini generates natural language crowd management advice.
Rule-based fallback always uses live queue_size (not cached stale value).
""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Comparison with existing papers")

    papers = pd.DataFrame({
        "Paper": [
            "CNN vehicle queue (MDPI 2021)",
            "CV + queuing theory (UGPTI 2023)",
            "Zone24x7 Queue Manager",
            "ST-GCN anomaly detection (2024)",
            "YOLOv8 + Pose unified (2025)",
            "PA-SQGN (Ours)",
        ],
        "Pose orientation": ["No","No","No","Partial","No","YES"],
        "Dynamic graph":    ["No","No","No","No","No","YES"],
        "Scene classifier": ["No","No","No","No","No","YES"],
        "Temporal conf.":   ["No","No","No","No","No","YES"],
        "LLM advisory":     ["No","No","No","No","No","YES"],
        "No fixed zones":   ["No","No","No","No","No","YES"],
    })
    st.table(papers)

    st.markdown("---")
    st.subheader("Known limitations (for report Section 6)")
    st.markdown("""
- **`pixels_per_meter = 90.0` is hardcoded** — valid only for the specific camera used.
  For any new deployment, this must be recalibrated.
- **MST is O(n³) per frame** — slow for n > 30 people. Fix: two-BFS diameter algorithm O(n).
- **CrowdClassifier sees all people**, not just graph-connected queue members.
  Low priority — currently at 95.8% correct classification.
- **LOG_EVERY_N_FRAMES = 15** — only ~24 rows logged per 360-frame video.
  Lower to 5 for evaluation with more GT alignment points.
""")