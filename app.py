from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import CATEGORIES, CATEGORY_DETECTORS, HybridInspectionEngine, blend_heatmap

st.set_page_config(
    page_title="Sentinel · Hybrid Visual Inspection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Look & feel -------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
      .stApp { background: radial-gradient(ellipse at 12% 0%, rgba(22,163,140,.12), transparent 34%), #0b1020; color: #edf3ff; }
      [data-testid="stSidebar"] { background: #10182a; border-right: 1px solid rgba(148,163,184,.16); }
      [data-testid="stHeader"] { background: rgba(11,16,32,.85); }
      .hero { border: 1px solid rgba(94,234,212,.23); border-radius: 24px; padding: 28px 30px; background: linear-gradient(135deg, rgba(15,39,53,.95), rgba(18,27,47,.98) 55%, rgba(42,26,59,.92)); margin-bottom: 20px; box-shadow: 0 18px 60px rgba(0,0,0,.22); }
      .eyebrow { font-family: 'DM Mono', monospace; letter-spacing: .14em; text-transform: uppercase; color: #5eead4; font-size: .76rem; }
      .hero h1 { font-size: clamp(2.2rem,4vw,3.35rem); line-height: 1.06; margin: 10px 0 12px; color: #f8fafc; font-weight: 800; letter-spacing: -.045em; }
      .hero p { color: #b9c8da; max-width: 880px; font-size: 1.02rem; }
      .chip { display:inline-block; padding: 6px 11px; border:1px solid rgba(148,163,184,.23); border-radius:99px; color:#cbd5e1; font-size:.78rem; margin: 4px 5px 0 0; background:rgba(15,23,42,.55); }
      .section-kicker { color:#5eead4; text-transform:uppercase; letter-spacing:.12em; font-family:'DM Mono',monospace; font-size:.73rem; margin-bottom:5px; }
      .panel { border:1px solid rgba(148,163,184,.18); border-radius:16px; background:rgba(17,26,45,.78); padding:16px 18px; }
      .muted { color:#9fb0c7; font-size:.88rem; }
      .metric-card { border:1px solid rgba(148,163,184,.18); border-radius:16px; padding:16px 18px; background:linear-gradient(135deg,rgba(19,32,52,.9),rgba(14,21,38,.9)); min-height:105px; }
      .metric-label { color:#9fb0c7; font-size:.82rem; }
      .metric-value { color:#f8fafc; font-weight:800; font-size:1.8rem; letter-spacing:-.04em; margin-top:4px; }
      div[data-testid="stFileUploader"] { border: 1px dashed rgba(94,234,212,.36); border-radius:16px; background:rgba(15,23,42,.55); padding:8px; }
      .stButton>button[kind="primary"] { border:0; border-radius:12px; background:linear-gradient(100deg,#0ea5a4,#4f46e5); color:white; font-weight:800; min-height:3rem; box-shadow:0 8px 28px rgba(20,184,166,.16); }
      .stDownloadButton>button { border-radius:10px; font-weight:700; }
      code, .mono { font-family:'DM Mono',monospace; }
      hr { border-color:rgba(148,163,184,.18); }
      [data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.18); border-radius:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Hybrid anomaly intelligence · video lab</div>
      <h1>Sentinel<span style="color:#5eead4">.</span><br>See the defect. Decide the response.</h1>
      <p>Upload an inspection clip and get frame-level anomaly scores, visual heatmaps, temporal risk estimates, and an RL-assisted triage decision — all from the models trained in your MVTec hybrid project.</p>
      <span class="chip">STFPM for grid</span><span class="chip">YOLO26 + PatchCore for other objects</span><span class="chip">Causal Temporal Transformer</span><span class="chip">Recall-balanced PPO V4</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Sidebar controls -------------------------
st.sidebar.markdown("## 🛠️ Inspection setup")
default_project = os.environ.get("MVTEC_PROJECT_DIR", str(Path.cwd()))
project_dir = st.sidebar.text_input(
    "Model project folder",
    value=default_project,
    help="Folder containing models/ and assets/ exported from your Google Drive project.",
)
device_option = st.sidebar.selectbox("Compute device", ["auto", "cuda", "cpu"], index=0)
category = st.sidebar.selectbox("Object category", CATEGORIES, index=0, help="Choose the object type being inspected. The hybrid model route depends on this choice.")
sampling_interval = st.sidebar.select_slider(
    "Sample one frame every",
    options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    value=0.25,
    format_func=lambda v: f"{v:g} sec",
    help="PPO V4 was trained at 4 decisions/second. Use 0.25 sec sampling to match its training cadence as closely as possible.",
)
max_frames = st.sidebar.slider("Maximum sampled frames", min_value=24, max_value=360, value=180, step=12)
enable_subtypes = st.sidebar.checkbox("Try defect-subtype classification", value=False, help="Optional DINOv2 + LinearSVM output for frames with high anomaly scores. Requires the saved DINOv2 backbone and category classifier; it may download DINOv2 source code on first use.")
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='muted'>Recommended: use a GPU for long videos. The input video is processed on the machine running this app; it is not uploaded to an external inference service.</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='muted'>Research demo, not a certified safety inspection system.</div>", unsafe_allow_html=True)

# ------------------------- Model readiness -------------------------
project_path = Path(project_dir).expanduser()
models_path = project_path / "models"
assets_path = project_path / "assets"
base_requirements = [
    models_path / "yolo26n.pt",
    models_path / "wide_resnet50_2_patch_extractor.pth",
    models_path / "stfpm_teacher_resnet18_grid.pth",
    models_path / "stfpm_student_resnet18_grid.pth",
    models_path / "causal_temporal_transformer.pt",
    models_path / "temporal_transformer_input_scaler.joblib",
    assets_path / "stfpm_calibration_grid.npz",
]
v4_policy_file = models_path / "ppo_monitor_flag_escalate_recall_v4toothbrush.zip"
legacy_policy_file = models_path / "ppo_inspection_policy.zip"
if not v4_policy_file.is_file() and not legacy_policy_file.is_file():
    # The UI can explain the exact missing V4 artifact instead of silently using an unrelated model.
    base_requirements.append(v4_policy_file)
for cat in CATEGORIES:
    base_requirements.extend([
        models_path / f"yolo26_embedding_knn_{cat}.joblib",
        models_path / f"patchcore_patch_bank_{cat}.pt",
        assets_path / f"calibration_{cat}.npz",
    ])
missing_paths = [p for p in base_requirements if not p.is_file()]

if missing_paths:
    st.warning(
        "The model bundle is not complete at the selected path. Export the trained `models/` and `assets/` folders from Google Drive, unzip them into one project folder, then point the sidebar to that folder."
    )
    with st.expander("Missing required artifacts"):
        for p in missing_paths:
            st.code(str(p), language="text")
    st.info("See README.md in the app bundle for the one-cell Google Colab export instructions.")
    engine = None
else:
    @st.cache_resource(show_spinner="Loading the hybrid detector, Transformer and PPO…")
    def load_engine_cached(path: str, device: str):
        return HybridInspectionEngine(path, device=device)

    engine = None
    try:
        engine = load_engine_cached(str(project_path.resolve()), device_option)
    except Exception as exc:
        st.error("Model loading failed. Check the exported artifacts and dependency versions.")
        st.exception(exc)

if engine is not None:
    if engine.ppo_mode == "v4_recall_balanced":
        st.success(f"Active controller: PPO V4 recall-balanced · {engine.ppo_model_filename} · learned actions, no rule-based override")
    else:
        st.warning(f"V4 checkpoint not found. Active controller is legacy PPO ({engine.ppo_model_filename}). Copy ppo_monitor_flag_escalate_recall_v4toothbrush.zip into models/ to activate V4.")

# ------------------------- Upload and inference -------------------------
left, right = st.columns([1.35, 0.65], gap="large")
with left:
    st.markdown("<div class='section-kicker'>01 / Upload</div>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader(
        "Drop an inspection video here",
        type=["mp4", "mov", "avi", "mkv", "mpeg", "mpg"],
        help="Short clips are best for the first run. Sampling is controlled in the sidebar.",
        label_visibility="collapsed",
    )
with right:
    st.markdown("<div class='section-kicker'>02 / Active route</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()}</div><div style='color:#5eead4;margin-top:6px;font-weight:700'>{CATEGORY_DETECTORS[category]}</div><div class='muted' style='margin-top:9px'>Frames → calibrated evidence → recall-balanced PPO V4 action</div></div>",
        unsafe_allow_html=True,
    )

if category == "grid":
    st.caption("This category is routed to the trained STFPM ResNet-18 student–teacher detector.")
else:
    st.caption("This category is routed to your trained YOLO26 global embedding + PatchCore-style local patch detector.")


def resize_for_display(frame: np.ndarray, max_w=720, max_h=480) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_w / max(w, 1), max_h / max(h, 1))
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def make_annotated_panel(original_bgr: np.ndarray, heat_bgr: np.ndarray, row: dict, category_name: str) -> np.ndarray:
    panel_w, panel_h = 420, 236
    def fit(img):
        img = cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        return img
    original = fit(original_bgr)
    heat = fit(heat_bgr)
    info = np.zeros((panel_h, 420, 3), dtype=np.uint8)
    info[:] = (25, 31, 48)
    action = str(row.get("ppo_final_action", "—"))
    score = float(row.get("anomaly_score", 0.0))
    tfp = float(row.get("temporal_anomaly_probability", 0.0))
    delta = float(row.get("score_delta", 0.0))
    if action == "HUMAN_REVIEW":
        action_color = (70, 80, 255)
    elif action == "FLAG_ANOMALY":
        action_color = (20, 170, 255)
    else:
        action_color = (95, 210, 155)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(info, "INSPECTION DECISION", (18, 29), font, 0.58, (160, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(info, f"{category_name.upper()}  |  {row.get('detector', '')[:27]}", (18, 58), font, 0.52, (235, 240, 250), 1, cv2.LINE_AA)
    cv2.putText(info, f"ANOMALY SCORE   {score:.3f}", (18, 96), font, 0.7, (70, 230, 205), 2, cv2.LINE_AA)
    cv2.putText(info, f"Temporal risk   {tfp:.3f}", (18, 130), font, 0.56, (225, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(info, f"Score delta     {delta:+.3f}", (18, 158), font, 0.56, (225, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(info, action, (18, 204), font, 0.66, action_color, 2, cv2.LINE_AA)
    combined = np.hstack([original, heat, info])
    header = np.zeros((44, combined.shape[1], 3), dtype=np.uint8)
    header[:] = (10, 16, 30)
    cv2.putText(header, f"SENTINEL   /   FRAME {row.get('frame_index', '')}   /   t={float(row.get('timestamp_sec', 0)):.2f}s",
                (16, 29), font, 0.6, (245, 248, 255), 1, cv2.LINE_AA)
    footer = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
    footer[:] = (10, 16, 30)
    cv2.putText(footer, "HEATMAP IS RELATIVE VISUAL EVIDENCE  ·  PPO OUTPUT IS EXPERIMENTAL TRIAGE",
                (16, 23), font, 0.47, (150, 175, 195), 1, cv2.LINE_AA)
    return np.vstack([header, combined, footer])


def process_video_file(video_bytes: bytes, original_name: str, active_engine: HybridInspectionEngine,
                       active_category: str, interval_sec: float, frame_limit: int,
                       subtype_predictions: bool):
    workdir = Path(tempfile.mkdtemp(prefix="sentinel_inspection_"))
    video_path = workdir / (Path(original_name).stem + Path(original_name).suffix.lower())
    video_path.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open this video. Try MP4 (H.264) or AVI.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if frame_count > 0 else None
    stride = max(1, int(round(fps * interval_sec)))
    if frame_count > 0:
        stride = max(stride, int(np.ceil(frame_count / frame_limit)))
    effective_interval = stride / fps
    if active_engine.ppo_mode == "v4_recall_balanced" and abs(effective_interval - 0.25) > 0.05:
        st.warning(
            f"PPO V4 was trained at 4 decisions/sec (0.25 sec cadence), but this clip will be sampled every "
            f"{effective_interval:.2f} sec because of the selected interval/frame budget. Temporal feature timing will differ from training. "
            "Increase the frame budget or use a shorter clip for the closest match."
        )
    rows = []
    meta = []
    frame_idx = 0
    sample_idx = 0
    progress = st.progress(0, text="Reading video frames…")
    status = st.empty()
    try:
        while sample_idx < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0 and sample_idx < frame_limit:
                status.caption(f"Scoring sampled frame {sample_idx + 1} · source frame {frame_idx}")
                original_shape = frame.shape[:2]
                frame_for_model = frame
                # Preserve visual detail for inference; save a resized copy for result rendering.
                scored = active_engine.score_frame(frame_for_model, active_category)
                display_frame = resize_for_display(frame, max_w=720, max_h=480)
                heat_overlay = blend_heatmap(display_frame, scored["heatmap"])
                source_path = workdir / f"frame_{sample_idx:05d}.jpg"
                heat_path = workdir / f"heat_{sample_idx:05d}.jpg"
                cv2.imwrite(str(source_path), display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                cv2.imwrite(str(heat_path), heat_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                row = {
                    "frame_index": int(frame_idx),
                    "sample_index": int(sample_idx),
                    "timestamp_sec": round(frame_idx / fps, 4),
                    "category": active_category,
                    "detector": scored["detector"],
                    "raw_score": float(scored["raw_score"]),
                    "anomaly_score": float(scored["anomaly_score"]),
                    "global_evidence_pct": float(scored["global_evidence_pct"]),
                    "local_evidence_pct": float(scored["local_evidence_pct"]),
                    "evidence_dispersion_pct": float(scored["evidence_dispersion_pct"]),
                    "evidence_max_pct": float(scored["evidence_max_pct"]),
                    "stfpm_or_patch_details": json.dumps(scored.get("details", {})),
                    "source_frame_path": str(source_path),
                    "heat_overlay_path": str(heat_path),
                }
                rows.append(row)
                meta.append({"source_shape": original_shape, "workdir": str(workdir)})
                sample_idx += 1
                if frame_count > 0:
                    progress.progress(min(0.70, 0.70 * frame_idx / max(frame_count, 1)), text="Scoring frames…")
            frame_idx += 1
    finally:
        cap.release()
    if not rows:
        shutil.rmtree(workdir, ignore_errors=True)
        raise RuntimeError("No frames were sampled from the uploaded video.")
    status.caption("Building temporal evidence and PPO decisions…")
    progress.progress(0.75, text="Running temporal Transformer + PPO…")
    temporal_rows = active_engine.temporal_and_rl(rows)

    if subtype_predictions:
        status.caption("Running optional subtype classifier on flagged frames…")
        for item in temporal_rows:
            item["predicted_subtype"] = ""
            if float(item["anomaly_score"]) >= 0.5 and item["ppo_final_action"] in {"FLAG_ANOMALY", "HUMAN_REVIEW"}:
                img = cv2.imread(item["source_frame_path"])
                if img is None:
                    continue
                try:
                    result = active_engine.predict_subtype(img, active_category)
                    if result:
                        item["predicted_subtype"] = result["predicted_subtype"]
                        item["subtype_decision_margin"] = result.get("decision_margin")
                except Exception as exc:
                    item["predicted_subtype"] = f"classifier unavailable: {str(exc)[:120]}"
                    break

    progress.progress(0.82, text="Rendering annotated inspection video…")
    processed_video_path = workdir / "sentinel_annotated_inspection.mp4"
    video_writer = None
    output_fps = float(np.clip(1.0 / max(effective_interval, 1.0 / fps), 0.5, 12.0))
    try:
        for index, row in enumerate(temporal_rows):
            frame_bgr = cv2.imread(row["source_frame_path"])
            heat_bgr = cv2.imread(row["heat_overlay_path"])
            if frame_bgr is None or heat_bgr is None:
                continue
            annotated = make_annotated_panel(frame_bgr, heat_bgr, row, active_category)
            if video_writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(processed_video_path), fourcc, output_fps, (w, h))
                if not video_writer.isOpened():
                    raise RuntimeError("Could not initialize MP4 writer. Check OpenCV/FFmpeg support.")
            video_writer.write(annotated)
            if index % max(1, len(temporal_rows) // 10) == 0:
                progress.progress(0.82 + 0.17 * (index + 1) / len(temporal_rows), text="Rendering annotated video…")
    finally:
        if video_writer is not None:
            video_writer.release()

    if not processed_video_path.is_file() or processed_video_path.stat().st_size < 1000:
        raise RuntimeError("The annotated video did not render correctly.")

    # Keep the compact sampled-frame previews in the temp folder for the in-app frame explorer.
    # They are not copied to Drive or included in the user-facing CSV.

    clean_rows = []
    for item in temporal_rows:
        clean = {k: v for k, v in item.items() if k not in {"heatmap", "rl_observation"}}
        clean_rows.append(clean)
    frame_table = pd.DataFrame(clean_rows)
    # Keep internal local paths out of user-facing reports.
    frame_table = frame_table.drop(columns=[c for c in ["source_frame_path", "heat_overlay_path"] if c in frame_table.columns])

    # Generate a self-contained downloadable results pack.
    csv_bytes = frame_table.to_csv(index=False).encode("utf-8")
    report = {
        "video_name": original_name,
        "category": active_category,
        "detector": CATEGORY_DETECTORS[active_category],
        "source_fps": fps,
        "source_frame_count": frame_count,
        "source_duration_seconds": duration_sec,
        "sampled_frames": len(frame_table),
        "requested_sampling_interval_seconds": interval_sec,
        "effective_sampling_interval_seconds": effective_interval,
        "anomaly_score_threshold_for_summary": 0.95,
        "frames_at_or_above_0_95": int((frame_table.anomaly_score >= 0.95).sum()),
        "ppo_action_counts": frame_table.ppo_final_action.value_counts().to_dict(),
        "ppo_policy_mode": active_engine.ppo_mode,
        "ppo_checkpoint": active_engine.ppo_model_filename,
        "human_review_event_count": int(
            frame_table["human_review_event"].sum()
            if "human_review_event" in frame_table.columns
            else (frame_table["ppo_final_action"] == "HUMAN_REVIEW").sum()
        ),
        "guardrail_reviews": int(frame_table.human_review_guardrail.sum()),
        "mean_anomaly_score": float(frame_table.anomaly_score.mean()),
        "max_anomaly_score": float(frame_table.anomaly_score.max()),
        "mean_temporal_anomaly_probability": float(frame_table.temporal_anomaly_probability.mean()),
        "research_caveat": "PPO V4 and the Temporal Transformer were trained/evaluated on synthetic sequences derived from MVTec still images, not continuous production video. The Transformer probability is displayed independently; V4 decisions use its trained 20-feature score/evidence/history observation. Video decisions are experimental triage outputs.",
    }
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    results_zip = workdir / "sentinel_inspection_results.zip"
    with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(processed_video_path, arcname=processed_video_path.name)
        zf.writestr("frame_level_report.csv", csv_bytes)
        zf.writestr("summary.json", report_bytes)
    progress.progress(1.0, text="Inspection complete.")
    status.success("Inspection finished. Review the frame timeline and download your outputs below.")
    return {
        "workdir": workdir,
        "video_path": processed_video_path,
        "csv_bytes": csv_bytes,
        "report_bytes": report_bytes,
        "zip_path": results_zip,
        "report": report,
        "table": frame_table,
        "rows": temporal_rows,
        "meta": meta,
        "output_fps": output_fps,
    }


if uploaded_video is not None:
    st.video(uploaded_video)
    video_mb = len(uploaded_video.getbuffer()) / (1024 ** 2)
    info_cols = st.columns(4)
    info_cols[0].metric("Video file", uploaded_video.name[:22])
    info_cols[1].metric("Upload size", f"{video_mb:.1f} MB")
    info_cols[2].metric("Sampling", f"every {sampling_interval:g}s")
    info_cols[3].metric("Frame budget", str(max_frames))
    run_clicked = st.button("🚀  Inspect this video", type="primary", use_container_width=True, disabled=engine is None)
    if run_clicked:
        with st.spinner("Starting the hybrid inspection pipeline…"):
            try:
                result = process_video_file(
                    uploaded_video.getvalue(), uploaded_video.name, engine, category,
                    sampling_interval, max_frames, enable_subtypes,
                )
                st.session_state["sentinel_result"] = result
                st.session_state["sentinel_category"] = category
            except Exception as exc:
                st.error("Video inspection failed. The exception below identifies the failing component.")
                st.exception(exc)

# Render saved result after reruns or download clicks.
result = st.session_state.get("sentinel_result")
if result is not None:
    st.markdown("---")
    st.markdown("<div class='section-kicker'>03 / Inspection report</div>", unsafe_allow_html=True)
    rpt = result["report"]
    actions = rpt.get("ppo_action_counts", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames analysed", f"{rpt['sampled_frames']:,}")
    c2.metric("Mean anomaly score", f"{rpt['mean_anomaly_score']:.3f}")
    c3.metric("Peak anomaly score", f"{rpt['max_anomaly_score']:.3f}")
    c4.metric("Frames ≥ 0.95", f"{rpt['frames_at_or_above_0_95']:,}")
    c5.metric("Human-review events", f"{rpt.get('human_review_event_count', actions.get('HUMAN_REVIEW', 0)):,}")

    chart = result["table"].set_index("timestamp_sec")[["anomaly_score", "temporal_anomaly_probability"]].rename(columns={
        "anomaly_score": "Detector score",
        "temporal_anomaly_probability": "Temporal probability",
    })
    st.markdown("### Risk timeline")
    st.line_chart(chart, height=260)
    st.caption("The detector score is a calibrated normal-reference percentile, not a defect probability. With PPO V4 active, the Transformer probability is displayed for context but is not fed to that policy.")

    tab_video, tab_frames, tab_actions, tab_notes = st.tabs(["🎞 Annotated video", "🧩 Frame explorer", "🧠 PPO decisions", "ℹ️ Interpretation"])
    with tab_video:
        st.video(result["video_path"].read_bytes())
        st.download_button(
            "⬇ Download annotated video (MP4)",
            data=result["video_path"].read_bytes(),
            file_name="sentinel_annotated_inspection.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
        if result["zip_path"].is_file():
            st.download_button(
                "⬇ Download complete inspection pack (MP4 + CSV + summary)",
                data=result["zip_path"].read_bytes(),
                file_name="sentinel_inspection_results.zip",
                mime="application/zip",
                use_container_width=True,
            )
    with tab_frames:
        table = result["table"]
        frame_i = st.slider("Choose analysed frame", 0, max(0, len(table)-1), 0, key="frame_explorer_idx")
        row = result["rows"][frame_i]
        frame_path = Path(row.get("source_frame_path", ""))
        heat_path = Path(row.get("heat_overlay_path", ""))
        # The internal frame cache gets pruned after rendering. Reconstruct a preview from the encoded video if needed.
        if frame_path.exists() and heat_path.exists():
            a, b = st.columns(2)
            a.image(cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB), caption="Sampled input frame", use_container_width=True)
            b.image(cv2.cvtColor(cv2.imread(str(heat_path)), cv2.COLOR_BGR2RGB), caption="Anomaly heatmap overlay", use_container_width=True)
        else:
            st.info("Source previews were cleaned after rendering to save disk space. Use the annotated MP4 for frame-by-frame review.")
        visible = table.iloc[frame_i]
        st.json({k: (None if pd.isna(v) else v) for k, v in visible.to_dict().items() if k not in ["stfpm_or_patch_details"]})
    with tab_actions:
        view_cols = ["timestamp_sec", "frame_index", "anomaly_score", "temporal_anomaly_probability", "score_delta", "ppo_raw_action", "ppo_final_action", "human_review_event", "review_cooldown_active_before_action", "sharp_rise_signal", "persistent_high_5s_signal", "escalation_candidate_signal", "human_review_guardrail", "human_review_guardrail_reason", "high_score_count_last_5"]
        view_cols = [c for c in view_cols if c in result["table"].columns]
        optional_cols = [c for c in ["predicted_subtype"] if c in result["table"].columns]
        st.dataframe(result["table"][view_cols + optional_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download frame-level CSV",
            data=result["csv_bytes"],
            file_name="sentinel_frame_level_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with tab_notes:
        st.markdown(
            """
            **How to read the output**
            - **Anomaly score:** empirical percentile relative to normal calibration images. Higher means more unusual; it is not a defect probability.
            - **Heatmap:** local model evidence. Color intensity is normalized per frame for visualization, so compare numeric scores rather than heatmap colors across frames.
            - **Temporal probability:** the trained Temporal Transformer estimate, calculated from recent score/evidence history.
            - **PPO action:** when the V4 checkpoint is present, this is the standalone recall-balanced PPO's direct action (`MONITOR`, `FLAG_ANOMALY`, or `HUMAN_REVIEW`). V4's 20-feature observation includes score jumps, rolling score statistics, high-score persistence, evidence percentiles, previous actions, and review cooldown. The app does not override V4 actions with a rule-based escalation guardrail. The Transformer probability remains a separate context signal.
            - **Subtype estimate:** optional DINOv2 + LinearSVM prediction. It is only a candidate label, not a verified diagnosis; the classifier was trained on MVTec defect images, not arbitrary industrial video.

            **Research limitation:** the Temporal Transformer and PPO were trained using synthetic sequences derived from static MVTec AD images. The app runs them on real video chronology, which is an experiment—not evidence of validated real-time video performance. Confirm important findings with a human inspector.
            """
        )

else:
    st.markdown("---")
    st.markdown("<div class='section-kicker'>How it works</div>", unsafe_allow_html=True)
    step1, step2, step3, step4 = st.columns(4)
    step1.markdown("<div class='panel'><b>01 · Detect</b><br><span class='muted'>STFPM for grid; YOLO26 + local PatchCore-style k-NN for the other four categories.</span></div>", unsafe_allow_html=True)
    step2.markdown("<div class='panel'><b>02 · Localize</b><br><span class='muted'>Frame-level anomaly heatmap and calibrated score.</span></div>", unsafe_allow_html=True)
    step3.markdown("<div class='panel'><b>03 · Remember</b><br><span class='muted'>Causal Transformer reads rolling evidence and sudden score changes.</span></div>", unsafe_allow_html=True)
    step4.markdown("<div class='panel'><b>04 · Triage</b><br><span class='muted'>PPO proposes do nothing, flag anomaly, or request human review.</span></div>", unsafe_allow_html=True)

st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#73849b;font-size:.78rem'><span>SENTINEL · HYBRID VISUAL INSPECTION</span><span>Local inference · Model routing from your MVTec project · Human review remains in the loop</span></div>", unsafe_allow_html=True)
