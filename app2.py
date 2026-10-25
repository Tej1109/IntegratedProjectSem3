from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
import html as html_lib
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference2 import (
    ANOMALY_DECISION_THRESHOLD, BUILTIN_CATEGORY_THRESHOLDS, CATEGORIES, CATEGORY_DETECTORS,
    DEFAULT_THRESHOLD_SELECTION_METHOD, DEFAULT_THRESHOLD_SELECTION_PARTITION,
    HybridInspectionEngine, blend_heatmap,
)

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
      <p>One inspection workspace, two routes: run the calibrated visual detector alone, or let the recall-balanced PPO V4 turn temporal evidence into inspection actions. Every high-score frame, decision and review event is traceable.</p>
      <span class="chip">▦ STFPM · grid</span><span class="chip">⛓ YOLO26 + PatchCore · zipper</span><span class="chip">◉ hazelnut</span><span class="chip">▤ leather</span><span class="chip">⌁ toothbrush</span><span class="chip">RL optional · PPO V4</span>
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
inspection_mode = st.sidebar.radio(
    "Inspection mode",
    ["Anomaly detection only", "RL-assisted inspection"],
    index=1,
    help="Anomaly detection only runs the visual detector and heatmaps. RL-assisted inspection additionally runs the temporal model and trained PPO policy.",
)
enable_rl = inspection_mode == "RL-assisted inspection"
category = st.sidebar.selectbox("Object category", CATEGORIES, index=0, help="Choose the object type being inspected. The hybrid model route depends on this choice.")
# Use the category thresholds from the last controller_dev F2 calibration run.
# The JSON generated by Colab overrides the bundled snapshot when present.
threshold_config_path = Path(project_dir).expanduser() / "assets" / "decision_thresholds.json"
threshold_config = {}
threshold_file_read_ok = False
try:
    if threshold_config_path.is_file():
        with threshold_config_path.open("r", encoding="utf-8") as _tf:
            threshold_config = json.load(_tf)
        threshold_file_read_ok = True
except Exception as _threshold_exc:
    st.sidebar.error(f"Could not read category decision thresholds: {_threshold_exc}")
    st.sidebar.warning("The inference engine refuses to use a malformed threshold file. Fix or remove decision_thresholds.json; removing it enables the bundled category thresholds.")

_threshold_map = dict(BUILTIN_CATEGORY_THRESHOLDS)
_file_threshold_map = threshold_config.get("thresholds", {}) if isinstance(threshold_config, dict) else {}
_valid_file_threshold_categories = set()
if isinstance(_file_threshold_map, dict):
    for _cat, _value in _file_threshold_map.items():
        if _cat in CATEGORIES:
            try:
                _value = float(_value)
                if np.isfinite(_value) and 0.0 <= _value <= 1.0:
                    _threshold_map[_cat] = _value
                    _valid_file_threshold_categories.add(_cat)
            except (TypeError, ValueError):
                pass

_has_category_threshold = category in _valid_file_threshold_categories
decision_boundary = float(_threshold_map.get(category, ANOMALY_DECISION_THRESHOLD))
if _has_category_threshold:
    _selection_method = threshold_config.get("selection_method", threshold_config.get("policy", DEFAULT_THRESHOLD_SELECTION_METHOD))
    _threshold_description = "loaded from assets/decision_thresholds.json"
    _threshold_details = f"Selected on `{threshold_config.get('selected_on', DEFAULT_THRESHOLD_SELECTION_PARTITION)}`."
elif category in BUILTIN_CATEGORY_THRESHOLDS:
    _selection_method = DEFAULT_THRESHOLD_SELECTION_METHOD
    _threshold_description = "bundled calibration values"
    _threshold_details = "These are category-specific thresholds selected on controller_dev in the latest calibration run. Copy the JSON from Drive to override them with your saved config."
else:
    _selection_method = "legacy compatibility fallback"
    _threshold_description = "legacy 0.95 fallback"
    _threshold_details = "No category-specific threshold is available."
st.sidebar.success(
    f"**Category decision policy**\n\n**{category} cutoff: {decision_boundary:.4f}**\n\n"
    f"Source: {_threshold_description}. {_threshold_details}\n\n"
    "The score stays continuous; this cutoff is used only for the ANOMALY/NORMAL label."
)
if not threshold_file_read_ok:
    st.sidebar.caption("The selected per-category thresholds are bundled in this app version. A valid assets/decision_thresholds.json overrides them.")
sampling_interval = st.sidebar.select_slider(
    "Sample one frame every",
    options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    value=0.25,
    format_func=lambda v: f"{v:g} sec",
    help="PPO V4 was trained on synthetic sequences at 4 decisions/second. 0.25 seconds best matches that cadence when the frame budget allows it.",
)
max_frames = st.sidebar.slider("Maximum sampled frames", min_value=24, max_value=360, value=180, step=12)
enable_subtypes = st.sidebar.checkbox("Try defect-subtype classification", value=False, help="Optional DINOv2 + LinearSVM output for frames with high anomaly scores. Requires the saved DINOv2 backbone and category classifier; it may download DINOv2 source code on first use.")

st.sidebar.markdown("### 🎯 Product region / ROI")
roi_mode = st.sidebar.radio(
    "Region sent to the detector",
    ["Center square crop (recommended)", "Full frame", "Custom ROI"],
    index=0,
    help="MVTec training images are square and tightly framed. Real video often includes background, borders, perspective, or empty space; cropping the product region can reduce domain-shift false alarms.",
)
roi_x_pct, roi_y_pct, roi_w_pct, roi_h_pct = 0, 0, 100, 100
if roi_mode == "Custom ROI":
    roi_x_pct = st.sidebar.slider("ROI left (%)", 0, 95, 10, 1)
    roi_y_pct = st.sidebar.slider("ROI top (%)", 0, 95, 10, 1)
    roi_w_pct = st.sidebar.slider("ROI width (%)", 5, 100, 80, 1)
    roi_h_pct = st.sidebar.slider("ROI height (%)", 5, 100, 80, 1)

st.sidebar.caption("If most frames score near 1.0, check this ROI, the selected category, camera lighting, and object scale before trusting the alerts.")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='muted'>Recommended: use a GPU for long videos. Video inference runs locally on the machine hosting this app.</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='muted'>MVTec-trained research prototype, not a certified safety inspection system.</div>", unsafe_allow_html=True)

# ------------------------- Model readiness -------------------------
project_path = Path(project_dir).expanduser()
models_path = project_path / "models"
assets_path = project_path / "assets"
base_requirements = [
    models_path / "yolo26n.pt",
    models_path / "wide_resnet50_2_patch_extractor.pth",
    models_path / "stfpm_teacher_resnet18_grid.pth",
    models_path / "stfpm_student_resnet18_grid.pth",
    assets_path / "stfpm_calibration_grid.npz",
]
if enable_rl:
    base_requirements.extend([
        models_path / "causal_temporal_transformer.pt",
        models_path / "temporal_transformer_input_scaler.joblib",
    ])
    v4_policy_file = models_path / "ppo_monitor_flag_escalate_recall_v4toothbrush.zip"
    legacy_policy_file = models_path / "ppo_inspection_policy.zip"
    if not v4_policy_file.is_file() and not legacy_policy_file.is_file():
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
        "The model bundle is incomplete at the selected project folder. Export your trained `models/` and `assets/` folders from Google Drive, then select the folder that contains those two subfolders."
    )
    with st.expander("Missing required artifacts"):
        for missing_path in missing_paths:
            st.code(str(missing_path), language="text")
    st.info("The model project path uses MVTEC_PROJECT_DIR when set; otherwise it defaults to the current working directory.")
    engine = None
else:
    def _threshold_file_signature(path: Path) -> str:
        try:
            stat = path.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return "missing"

    @st.cache_resource(show_spinner="Loading the visual detector and selected inspection route…")
    def load_engine_cached(path: str, device: str, use_rl: bool, threshold_signature: str):
        # threshold_signature is intentionally part of the cache key so changing the JSON
        # cannot leave an old threshold map inside a cached inference engine.
        return HybridInspectionEngine(path, device=device, enable_rl=use_rl)

    engine = None
    try:
        engine = load_engine_cached(
            str(project_path.resolve()), device_option, enable_rl,
            _threshold_file_signature(assets_path / "decision_thresholds.json"),
        )
    except Exception as exc:
        st.error("Model loading failed. Check that the project path and model dependencies match the exported bundle.")
        st.exception(exc)

if engine is not None:
    if not enable_rl:
        st.success("Active route: visual anomaly detection only · Transformer/PPO are not loaded for this run")
    elif engine.ppo_mode == "v4_recall_balanced":
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
        f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()}</div><div style='color:#5eead4;margin-top:6px;font-weight:700'>{CATEGORY_DETECTORS[category]}</div><div class='muted' style='margin-top:9px'>" + ("Frames → calibrated anomaly score → heatmap" if not enable_rl else "Frames → calibrated evidence → PPO V4 inspection action") + "</div></div>",
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


def compute_roi_bbox(frame_shape, mode: str, x_pct=0, y_pct=0, w_pct=100, h_pct=100):
    """Return (x1,y1,x2,y2) in source-frame pixels; end coordinates are exclusive."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h < 2 or w < 2:
        return (0, 0, max(w, 1), max(h, 1))
    if mode == "Center square crop (recommended)":
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        return (x1, y1, x1 + side, y1 + side)
    if mode == "Custom ROI":
        x1 = int(round(w * float(x_pct) / 100.0))
        y1 = int(round(h * float(y_pct) / 100.0))
        x2 = int(round(w * (float(x_pct) + float(w_pct)) / 100.0))
        y2 = int(round(h * (float(y_pct) + float(h_pct)) / 100.0))
        x1 = min(max(x1, 0), w - 1)
        y1 = min(max(y1, 0), h - 1)
        x2 = min(max(x2, x1 + 1), w)
        y2 = min(max(y2, y1 + 1), h)
        return (x1, y1, x2, y2)
    return (0, 0, w, h)


def blend_heatmap_in_roi(display_frame, heatmap, source_bbox, source_shape, alpha=0.48):
    """Map a crop-space heatmap back into the matching ROI of the display frame only."""
    display = display_frame.copy()
    dh, dw = display.shape[:2]
    sh, sw = int(source_shape[0]), int(source_shape[1])
    x1, y1, x2, y2 = source_bbox
    sx = dw / max(sw, 1)
    sy = dh / max(sh, 1)
    dx1 = int(round(x1 * sx)); dx2 = int(round(x2 * sx))
    dy1 = int(round(y1 * sy)); dy2 = int(round(y2 * sy))
    dx1 = min(max(dx1, 0), dw - 1); dx2 = min(max(dx2, dx1 + 1), dw)
    dy1 = min(max(dy1, 0), dh - 1); dy2 = min(max(dy2, dy1 + 1), dh)
    roi = display[dy1:dy2, dx1:dx2]
    heat_overlay = blend_heatmap(roi, heatmap, alpha=alpha)
    display[dy1:dy2, dx1:dx2] = heat_overlay
    # Explicitly show the evaluated region so users know background outside it was not scored.
    cv2.rectangle(display, (dx1, dy1), (dx2 - 1, dy2 - 1), (210, 230, 100), 1, cv2.LINE_AA)
    return display


def make_annotated_panel(
    original_bgr: np.ndarray,
    heat_bgr: np.ndarray,
    row: dict,
    category_name: str,
    inspection_mode: str,
    score_history: list[float],
    decision_boundary: float,
) -> np.ndarray:
    """Render source frame, the existing ROI-aligned heatmap, action/score and a score sparkline."""
    panel_w, panel_h = 420, 260

    def fit(img):
        return cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    original = fit(original_bgr)
    heat = fit(heat_bgr)
    info = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    info[:] = (25, 31, 48)
    score = float(row.get("anomaly_score", 0.0))
    is_rl = inspection_mode == "RL-assisted inspection"
    action = str(row.get("ppo_final_action", "DETECTION ONLY")) if is_rl else "DETECTION ONLY"
    tfp = row.get("temporal_anomaly_probability", np.nan)
    delta = float(row.get("score_delta", 0.0))
    if action == "HUMAN_REVIEW":
        action_color = (70, 80, 255)
    elif action == "FLAG_ANOMALY":
        action_color = (20, 170, 255)
    elif action == "MONITOR":
        action_color = (95, 210, 155)
    else:
        action_color = (190, 210, 220)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(info, "INSPECTION SUMMARY", (18, 28), font, 0.58, (160, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(info, f"{category_name.upper()}  |  {str(row.get('detector', ''))[:27]}", (18, 55), font, 0.48, (235, 240, 250), 1, cv2.LINE_AA)
    cv2.putText(info, f"ANOMALY SCORE   {score:.3f}", (18, 92), font, 0.68, (70, 230, 205), 2, cv2.LINE_AA)
    prediction = str(row.get("detector_prediction", "ANOMALY" if row.get("is_anomaly", score >= decision_boundary) else "NORMAL"))
    prediction_color = (55, 65, 255) if prediction == "ANOMALY" else (80, 220, 145)
    cv2.putText(info, f"MODEL PREDICTION: {prediction}", (18, 119), font, 0.51, prediction_color, 2, cv2.LINE_AA)
    if is_rl and pd.notna(tfp):
        cv2.putText(info, f"Temporal {float(tfp):.3f}  |  score delta {delta:+.3f}", (18, 141), font, 0.42, (225, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(info, action[:28], (18, 164), font, 0.60 if len(action) < 18 else 0.46, action_color, 2, cv2.LINE_AA)

    # In-video score trace; the notebook decision boundary is fixed and shown for context.
    graph_x, graph_y, graph_w, graph_h = 20, 190, 380, 42
    cv2.putText(info, f"ANOMALY SCORE · DECISION LINE {decision_boundary:.3f}", (graph_x, graph_y - 5), font, 0.37, (160, 185, 205), 1, cv2.LINE_AA)
    cv2.rectangle(info, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (75, 88, 110), 1, cv2.LINE_AA)
    threshold_y = graph_y + graph_h - int(np.clip(decision_boundary, 0, 1) * (graph_h - 8)) - 4
    cv2.line(info, (graph_x + 1, threshold_y), (graph_x + graph_w - 1, threshold_y), (50, 120, 225), 1, cv2.LINE_AA)
    values = np.asarray(score_history[-60:], dtype=np.float32)
    if values.size >= 1:
        points = []
        for i, val in enumerate(values):
            x = graph_x + 3 + int(i * (graph_w - 6) / max(len(values) - 1, 1))
            y = graph_y + graph_h - 4 - int(np.clip(float(val), 0, 1) * (graph_h - 8))
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(info, [np.asarray(points, dtype=np.int32)], False, (80, 230, 185), 2, cv2.LINE_AA)
        else:
            cv2.circle(info, points[0], 3, (80, 230, 185), -1, cv2.LINE_AA)
    combined = np.hstack([original, heat, info])
    header = np.zeros((44, combined.shape[1], 3), dtype=np.uint8)
    header[:] = (10, 16, 30)
    cv2.putText(
        header,
        f"SENTINEL / FRAME {row.get('frame_index', '')} / t={float(row.get('timestamp_sec', 0)):.2f}s",
        (16, 29), font, 0.60, (245, 248, 255), 1, cv2.LINE_AA,
    )
    footer = np.zeros((34, combined.shape[1], 3), dtype=np.uint8)
    footer[:] = (10, 16, 30)
    footer_text = "LOCALIZATION MAP IS FRAME-NORMALIZED · SCORE IS A NORMAL-REFERENCE PERCENTILE"
    if is_rl:
        footer_text = "PPO ACTIONS ARE EXPERIMENTAL TRIAGE · HUMAN CONFIRMATION REMAINS IMPORTANT"
    cv2.putText(footer, footer_text, (16, 22), font, 0.45, (150, 175, 195), 1, cv2.LINE_AA)
    return np.vstack([header, combined, footer])


def make_browser_compatible_video(source_path: Path, output_dir: Path):
    """Transcode OpenCV's mp4v output to browser-safe H.264 MP4 (or VP9 WebM fallback)."""
    ffmpeg = shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if ffmpeg:
        mp4_path = output_dir / "sentinel_annotated_inspection.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if completed.returncode == 0 and mp4_path.is_file() and mp4_path.stat().st_size > 1000:
                return mp4_path, "video/mp4", None
        except Exception:
            pass
        webm_path = output_dir / "sentinel_annotated_inspection.webm"
        webm_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path),
            "-an", "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "5",
            "-b:v", "0", "-crf", "31", str(webm_path),
        ]
        try:
            completed = subprocess.run(webm_cmd, capture_output=True, text=True, timeout=180)
            if completed.returncode == 0 and webm_path.is_file() and webm_path.stat().st_size > 1000:
                return webm_path, "video/webm", "H.264 export was unavailable, so browser playback uses WebM/VP9."
        except Exception:
            pass
    return source_path, "video/mp4", "Could not transcode this OpenCV MP4 to a browser-friendly codec. The raw MP4 is still downloadable; install imageio-ffmpeg and rerun to enable H.264 playback."


def svg_score_chart(table: pd.DataFrame, decision_boundary: float, width: int = 980, height: int = 260) -> str:
    """Small dependency-free score chart embedded in downloadable HTML reports."""
    values = pd.to_numeric(table.get("anomaly_score", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy()
    if values.size == 0:
        return "<p>No score samples available.</p>"
    left, right, top, bottom = 52, width - 22, 20, height - 36
    usable_w, usable_h = right - left, bottom - top
    points = []
    for i, value in enumerate(values):
        x = left + i * usable_w / max(len(values) - 1, 1)
        y = bottom - float(np.clip(value, 0, 1)) * usable_h
        points.append(f"{x:.1f},{y:.1f}")
    t_y = bottom - float(np.clip(decision_boundary, 0, 1)) * usable_h
    grid = []
    for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
        gy = bottom - tick * usable_h
        grid.append(
            f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#26374f" stroke-width="1"/>'
            f'<text x="8" y="{gy + 4:.1f}" fill="#9fb0c7" font-size="11">{tick:.2f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Anomaly score timeline with fixed zero-to-one axis" style="width:100%;height:auto">'
        f'<rect x="{left}" y="{top}" width="{usable_w}" height="{usable_h}" rx="8" fill="#101a2b" stroke="#2a3850"/>'
        + "".join(grid) +
        f'<line x1="{left}" y1="{t_y:.1f}" x2="{right}" y2="{t_y:.1f}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 5"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#5eead4" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<text x="{left}" y="{height - 8}" fill="#9fb0c7" font-size="12">first sampled frame</text>'
        f'<text x="{right - 130}" y="{height - 8}" fill="#9fb0c7" font-size="12">last sampled frame</text>'
        f'<text x="{right - 160}" y="{max(top + 13, t_y - 7):.1f}" fill="#fbbf24" font-size="12">threshold {decision_boundary:.4f}</text></svg>'
    )


def build_dashboard_html(report: dict, table: pd.DataFrame) -> bytes:
    """Build a portable, standalone HTML report/dashboard with summary, timeline and top frames."""
    def esc(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        return html_lib.escape(str(value))

    decision_boundary = float(report.get("decision_boundary", ANOMALY_DECISION_THRESHOLD))
    high_n = int(report.get("high_anomaly_frame_count", 0))
    n = int(report.get("sampled_frames", len(table)))
    max_row = report.get("max_score_location", {})
    min_row = report.get("min_score_location", {})
    is_rl = report.get("inspection_mode") == "RL-assisted inspection"
    metrics = [
        ("Frames sampled", f"{n:,}"),
        (f"Model-predicted anomalies", f"{high_n:,}"),
        ("Mean score", f"{float(report.get('mean_anomaly_score', 0)):.3f}"),
        ("Peak score", f"{float(report.get('max_anomaly_score', 0)):.3f}"),
        ("Lowest score", f"{float(report.get('min_anomaly_score', 0)):.3f}"),
    ]
    if is_rl:
        metrics += [("Human-review events", f"{int(report.get('human_review_event_count', 0)):,}"),
                    ("Frames labelled FLAG_ANOMALY", f"{int(report.get('flag_action_frames', 0)):,}")]
    cards = "".join(f'<div class="metric"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div></div>' for label, value in metrics)
    chart = svg_score_chart(table, decision_boundary)

    cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "detector_prediction", "is_anomaly", "ppo_final_action", "human_review_event"] if c in table.columns]
    if "anomaly_score" in table.columns:
        top = table.sort_values("anomaly_score", ascending=False).head(12)
    else:
        top = table.head(0)
    def rows_html(df, columns):
        body = []
        for _, rr in df[columns].iterrows():
            cells = []
            for col in columns:
                val = rr[col]
                if isinstance(val, (float, np.floating)) and np.isfinite(val):
                    if col in {"timestamp_sec"}:
                        val = f"{float(val):.2f}s"
                    elif col in {"anomaly_score"}:
                        val = f"{float(val):.3f}"
                    else:
                        val = f"{float(val):.3f}"
                cells.append(f"<td>{esc(val)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "".join(body) or f'<tr><td colspan="{max(1,len(columns))}">No rows</td></tr>'
    top_table = "<table><thead><tr>" + "".join(f"<th>{esc(c.replace('_',' ').title())}</th>" for c in cols) + "</tr></thead><tbody>" + rows_html(top, cols) + "</tbody></table>"
    action_section = ""
    if is_rl:
        counts = report.get("ppo_action_counts", {}) or {}
        bars = []
        max_count = max(counts.values(), default=1)
        for name in ["MONITOR", "FLAG_ANOMALY", "HUMAN_REVIEW", "DO_NOTHING"]:
            count = int(counts.get(name, 0))
            if name not in counts and count == 0:
                continue
            width_pct = 0 if max_count == 0 else (count / max_count * 100)
            bars.append(f'<div class="barrow"><div class="barlabel">{esc(name)}</div><div class="bartrack"><div class="bar" style="width:{width_pct:.1f}%"></div></div><div class="barvalue">{count:,}</div></div>')
        reviews = table[table.get("ppo_final_action", pd.Series(index=table.index, dtype=str)).astype(str).eq("HUMAN_REVIEW")] if "ppo_final_action" in table.columns else table.head(0)
        review_cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "human_review_event"] if c in reviews.columns]
        review_table = "<table><thead><tr>" + "".join(f"<th>{esc(c.replace('_',' ').title())}</th>" for c in review_cols) + "</tr></thead><tbody>" + rows_html(reviews.head(30), review_cols) + "</tbody></table>"
        action_section = f'<section><h2>RL action mix</h2>{"".join(bars) or "<p>No action counts available.</p>"}<h2>Human-review frames</h2>{review_table}</section>'

    caveat = esc(report.get("research_caveat", ""))
    meta = f"{esc(report.get('video_name','uploaded video'))} · {esc(report.get('category',''))} · {esc(report.get('inspection_mode',''))} · {esc(report.get('detector',''))}"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel Inspection Dashboard</title>
<style>
:root{{--bg:#0b1020;--panel:#111a2d;--stroke:#2a3850;--text:#edf3ff;--muted:#9fb0c7;--mint:#5eead4;--amber:#fbbf24}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 10% 0%,#123039 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1200px;margin:auto;padding:32px 24px 60px}}.eyebrow{{font:12px ui-monospace,monospace;letter-spacing:.15em;text-transform:uppercase;color:var(--mint)}}h1{{font-size:clamp(32px,5vw,54px);line-height:1.04;letter-spacing:-.04em;margin:10px 0}}.meta{{color:var(--muted);margin-bottom:26px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:22px 0}}.metric,section{{border:1px solid var(--stroke);background:rgba(17,26,45,.88);border-radius:16px;padding:18px}}.label{{color:var(--muted);font-size:13px}}.value{{font-size:28px;font-weight:800;color:#fff;margin-top:5px}}section{{margin-top:18px}}h2{{font-size:20px;margin:0 0 16px}}p,.note{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;font-size:13px;overflow-wrap:anywhere}}th,td{{padding:10px 9px;border-bottom:1px solid var(--stroke);text-align:left}}th{{color:var(--mint);font-weight:700}}.tablewrap{{overflow-x:auto}}.barrow{{display:grid;grid-template-columns:170px 1fr 55px;gap:12px;align-items:center;margin:14px 0}}.barlabel{{font:12px ui-monospace,monospace}}.bartrack{{background:#0b1020;border-radius:99px;height:13px;overflow:hidden;border:1px solid var(--stroke)}}.bar{{height:100%;background:linear-gradient(90deg,#0ea5a4,#6366f1);border-radius:99px}}.barvalue{{text-align:right;font-weight:700}}.foot{{font-size:12px;color:var(--muted);margin-top:24px}}@media(max-width:600px){{.barrow{{grid-template-columns:120px 1fr 40px}}main{{padding:20px 12px}}}}
</style></head><body><main><div class="eyebrow">Sentinel · Inspection intelligence</div><h1>Inspection dashboard<span style="color:var(--mint)">.</span></h1><div class="meta">{meta}</div><div class="metrics">{cards}</div>
<section><h2>Anomaly score timeline</h2>{chart}<p>Dashed amber line = this category’s decision cutoff. The y-axis is fixed from 0 to 1. The score is a normal-reference percentile, not a probability of defect.</p><p>{int(report.get('score_saturated_at_one_count', 0))} of {int(report.get('sampled_frames', 0))} sampled frames ({100.0 * float(report.get('score_saturated_at_one_rate', 0.0)):.1f}%) reached 1.000. ECDF calibration maps a score at or above the largest normal-reference score to 1.000; see the exported frame-level CSV’s pre_final_calibration_score column for the values before that mapping.</p></section>
<section><h2>Peak / minimum score location</h2><div class="tablewrap"><table><thead><tr><th>Statistic</th><th>Score</th><th>Source frame</th><th>Timestamp</th></tr></thead><tbody><tr><td>Maximum</td><td>{float(report.get('max_anomaly_score',0)):.3f}</td><td>{esc(max_row.get('frame_index','—'))}</td><td>{esc(max_row.get('timestamp_sec','—'))}</td></tr><tr><td>Minimum</td><td>{float(report.get('min_anomaly_score',0)):.3f}</td><td>{esc(min_row.get('frame_index','—'))}</td><td>{esc(min_row.get('timestamp_sec','—'))}</td></tr></tbody></table></div></section>
<section><h2>Highest-scoring sampled frames</h2><div class="tablewrap">{top_table}</div></section>
{action_section}
<section><h2>Interpretation & limitation</h2><p>{caveat}</p><p>Review the source frame and heatmap alongside the numeric score. A high score means unusual relative to the saved normal calibration reference; it does not prove a defect.</p></section>
<div class="foot">Generated by Sentinel · Local inference · Exported dashboard · {esc(report.get('video_name',''))}</div></main></body></html>"""
    return html.encode("utf-8")

def process_video_file(video_bytes: bytes, original_name: str, active_engine: HybridInspectionEngine,
                       active_category: str, interval_sec: float, frame_limit: int,
                       subtype_predictions: bool, roi_mode: str = "Center square crop (recommended)",
                       roi_x_pct: int = 0, roi_y_pct: int = 0,
                       roi_w_pct: int = 100, roi_h_pct: int = 100,
                       inspection_mode: str = "RL-assisted inspection",
                       decision_boundary: float = ANOMALY_DECISION_THRESHOLD):
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
    if inspection_mode == "RL-assisted inspection" and getattr(active_engine, "ppo_mode", "") == "v4_recall_balanced" and abs(effective_interval - 0.25) > 0.05:
        st.warning(
            f"PPO V4 was trained at 4 synthetic decisions/second; the effective sampled cadence here is {effective_interval:.2f}s because of the video length/frame budget. Treat temporal decisions as experimental."
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
                roi_bbox = compute_roi_bbox(
                    original_shape, roi_mode,
                    x_pct=roi_x_pct, y_pct=roi_y_pct,
                    w_pct=roi_w_pct, h_pct=roi_h_pct,
                )
                rx1, ry1, rx2, ry2 = roi_bbox
                frame_for_model = frame[ry1:ry2, rx1:rx2]
                if frame_for_model.size == 0 or min(frame_for_model.shape[:2]) < 16:
                    raise RuntimeError(
                        f"ROI is too small at source frame {frame_idx}: {roi_bbox}. "
                        "Increase the ROI width/height or use center/full frame."
                    )
                # Score only the selected object region; map the heatmap back to that
                # same region instead of stretching it over the whole camera frame.
                scored = active_engine.score_frame(frame_for_model, active_category)
                display_frame = resize_for_display(frame, max_w=720, max_h=480)
                heat_overlay = blend_heatmap_in_roi(
                    display_frame, scored["heatmap"], roi_bbox, original_shape
                )
                source_path = workdir / f"frame_{sample_idx:05d}.jpg"
                roi_path = workdir / f"roi_{sample_idx:05d}.jpg"
                heat_path = workdir / f"heat_{sample_idx:05d}.jpg"
                cv2.imwrite(str(source_path), display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                cv2.imwrite(str(roi_path), frame_for_model, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(str(heat_path), heat_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                row = {
                    "frame_index": int(frame_idx),
                    "sample_index": int(sample_idx),
                    "timestamp_sec": round(frame_idx / fps, 4),
                    "category": active_category,
                    "inspection_mode": inspection_mode,
                    "decision_boundary": float(scored.get("decision_boundary", decision_boundary)),
                    "decision_threshold_source": str(scored.get("decision_threshold_source", "")),
                    "is_anomaly": bool(scored.get("is_anomaly", float(scored["anomaly_score"]) >= decision_boundary)),
                    "detector_prediction": str(scored.get("detector_prediction", "ANOMALY" if float(scored["anomaly_score"]) >= decision_boundary else "NORMAL")),

                    "roi_mode": roi_mode,
                    "roi_x1": int(rx1), "roi_y1": int(ry1),
                    "roi_x2": int(rx2), "roi_y2": int(ry2),
                    "detector": scored["detector"],
                    "raw_score": float(scored["raw_score"]),
                    "pre_final_calibration_score": float(scored.get("pre_final_calibration_score", scored["raw_score"])),
                    "anomaly_score": float(scored["anomaly_score"]),
                    "calibration_reference_count": int(scored.get("calibration_reference_count", 0)),
                    "calibration_reference_max_raw_score": float(scored.get("calibration_reference_max_raw_score", np.nan)),
                    "score_saturated_at_one": bool(scored.get("score_saturated_at_one", float(scored["anomaly_score"]) >= 1.0)),
                    "global_evidence_pct": float(scored["global_evidence_pct"]),
                    "local_evidence_pct": float(scored["local_evidence_pct"]),
                    "evidence_dispersion_pct": float(scored["evidence_dispersion_pct"]),
                    "evidence_max_pct": float(scored["evidence_max_pct"]),
                    "stfpm_or_patch_details": json.dumps(scored.get("details", {})),
                    "source_frame_path": str(source_path),
                    "model_input_path": str(roi_path),
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
    if inspection_mode == "RL-assisted inspection":
        status.caption("Building temporal evidence and PPO decisions…")
        progress.progress(0.75, text="Running temporal Transformer + PPO V4…")
        temporal_rows = active_engine.temporal_and_rl(rows)
    else:
        status.caption("Preparing detection-only results…")
        progress.progress(0.75, text="Preparing detector-only frames…")
        temporal_rows = []
        score_history = []
        for i, row in enumerate(rows):
            item = dict(row)
            score = float(item["anomaly_score"])
            item["score_delta"] = float(score - score_history[-1]) if score_history else 0.0
            item["rolling_mean_score_5"] = float(np.mean((score_history + [score])[-5:]))
            item["rolling_std_score_5"] = float(np.std((score_history + [score])[-5:]))
            item["decision"] = str(item.get("detector_prediction", "ANOMALY" if bool(item.get("is_anomaly", score >= decision_boundary)) else "NORMAL"))
            temporal_rows.append(item)
            score_history.append(score)

    if subtype_predictions:
        status.caption("Running optional subtype classifier on model-predicted anomaly frames…")
        for item in temporal_rows:
            item["predicted_subtype"] = ""
            # Subtype diagnosis follows the detector prediction. PPO action remains a separate
            # decision about what to do next (monitor, flag, or human review).
            if bool(item.get("is_anomaly", float(item["anomaly_score"]) >= decision_boundary)):
                # Use the same cropped ROI that was scored by the anomaly detector.
                img_path = item.get("model_input_path") or item["source_frame_path"]
                img = cv2.imread(img_path)
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

    progress.progress(0.82, text="Rendering annotated inspection video with score timeline…")
    raw_video_path = workdir / "sentinel_annotated_inspection_raw.mp4"
    video_writer = None
    output_fps = float(np.clip(1.0 / max(effective_interval, 1.0 / fps), 0.5, 12.0))
    score_history_for_video = []
    try:
        for index, row in enumerate(temporal_rows):
            frame_bgr = cv2.imread(row["source_frame_path"])
            heat_bgr = cv2.imread(row["heat_overlay_path"])
            if frame_bgr is None or heat_bgr is None:
                continue
            score_history_for_video.append(float(row.get("anomaly_score", 0.0)))
            annotated = make_annotated_panel(
                frame_bgr, heat_bgr, row, active_category,
                inspection_mode=inspection_mode,
                score_history=score_history_for_video,
                decision_boundary=decision_boundary,
            )
            if video_writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(raw_video_path), fourcc, output_fps, (w, h))
                if not video_writer.isOpened():
                    raise RuntimeError("Could not initialize the OpenCV video writer. Try installing a full OpenCV/FFmpeg build.")
            video_writer.write(annotated)
            if index % max(1, len(temporal_rows) // 10) == 0:
                progress.progress(0.82 + 0.10 * (index + 1) / len(temporal_rows), text="Rendering annotated video…")
    finally:
        if video_writer is not None:
            video_writer.release()

    if not raw_video_path.is_file() or raw_video_path.stat().st_size < 1000:
        raise RuntimeError("The annotated video did not render correctly.")

    processed_video_path, video_mime_type, playback_warning = make_browser_compatible_video(raw_video_path, workdir)

    if not processed_video_path.is_file() or processed_video_path.stat().st_size < 1000:
        raise RuntimeError("The annotated video did not render correctly.")

    # Keep the compact sampled-frame previews in the temp folder for the in-app frame explorer.
    # They are not copied to Drive or included in the user-facing CSV.

    clean_rows = []
    for item in temporal_rows:
        clean = {k: v for k, v in item.items() if k not in {"heatmap", "rl_observation", "model_input_path"}}
        clean_rows.append(clean)
    frame_table = pd.DataFrame(clean_rows)
    frame_table = frame_table.drop(columns=[c for c in ["source_frame_path", "heat_overlay_path"] if c in frame_table.columns])
    if "is_anomaly" not in frame_table.columns:
        frame_table["is_anomaly"] = frame_table["anomaly_score"].astype(float) >= float(decision_boundary)
    else:
        frame_table["is_anomaly"] = frame_table["is_anomaly"].astype(bool)
    frame_table["detector_prediction"] = np.where(frame_table["is_anomaly"], "ANOMALY", "NORMAL")
    frame_table["decision_boundary"] = float(decision_boundary)

    csv_bytes = frame_table.to_csv(index=False).encode("utf-8")
    max_idx = int(frame_table["anomaly_score"].astype(float).idxmax())
    min_idx = int(frame_table["anomaly_score"].astype(float).idxmin())
    max_loc_row = frame_table.loc[max_idx]
    min_loc_row = frame_table.loc[min_idx]
    rl_enabled = inspection_mode == "RL-assisted inspection"
    action_counts = frame_table["ppo_final_action"].value_counts().to_dict() if rl_enabled and "ppo_final_action" in frame_table.columns else {}
    review_events = (
        int(frame_table["human_review_event"].sum())
        if rl_enabled and "human_review_event" in frame_table.columns
        else int(action_counts.get("HUMAN_REVIEW", 0)) if rl_enabled else 0
    )
    report = {
        "video_name": original_name,
        "category": active_category,
        "detector": CATEGORY_DETECTORS[active_category],
        "inspection_mode": inspection_mode,
        "roi_mode": roi_mode,
        "roi_custom_percentages": {
            "left": int(roi_x_pct), "top": int(roi_y_pct),
            "width": int(roi_w_pct), "height": int(roi_h_pct),
        } if roi_mode == "Custom ROI" else None,
        "source_fps": fps,
        "source_frame_count": frame_count,
        "source_duration_seconds": duration_sec,
        "sampled_frames": len(frame_table),
        "requested_sampling_interval_seconds": interval_sec,
        "effective_sampling_interval_seconds": effective_interval,
        "decision_boundary": float(decision_boundary),
        "decision_threshold_policy": getattr(active_engine, "decision_threshold_policy", "legacy fixed 0.95 fallback"),
        "decision_threshold_source": str(next((r.get("decision_threshold_source", "") for r in temporal_rows if r.get("decision_threshold_source")), "")),
        "high_anomaly_frame_count": int(frame_table["is_anomaly"].sum()),  # compatibility alias
        "predicted_anomaly_frame_count": int(frame_table["is_anomaly"].sum()),
        "frames_at_or_above_decision_threshold": int((frame_table.anomaly_score >= float(decision_boundary)).sum()),
        "frames_at_or_above_0_95": int((frame_table.anomaly_score >= 0.95).sum()),  # historical comparison, not the active cutoff
        "ppo_action_counts": action_counts,
        "human_review_event_count": review_events,
        "review_action_frames": int(action_counts.get("HUMAN_REVIEW", 0)),
        "flag_action_frames": int(action_counts.get("FLAG_ANOMALY", 0)),
        "monitor_action_frames": int(action_counts.get("MONITOR", action_counts.get("DO_NOTHING", 0))),
        "guardrail_reviews": int(frame_table["human_review_guardrail"].sum()) if "human_review_guardrail" in frame_table.columns else 0,
        "mean_anomaly_score": float(frame_table.anomaly_score.mean()),
        "median_anomaly_score": float(frame_table.anomaly_score.median()),
        "std_anomaly_score": float(frame_table.anomaly_score.std(ddof=0)),
        "score_saturated_at_one_count": int((frame_table["anomaly_score"].astype(float) >= 1.0 - 1e-9).sum()),
        "score_saturated_at_one_rate": float((frame_table["anomaly_score"].astype(float) >= 1.0 - 1e-9).mean()),
        "unique_calibrated_score_count": int(frame_table["anomaly_score"].astype(float).round(8).nunique()),
        "pre_final_score_min": float(pd.to_numeric(frame_table.get("pre_final_calibration_score", pd.Series(dtype=float)), errors="coerce").min()),
        "pre_final_score_max": float(pd.to_numeric(frame_table.get("pre_final_calibration_score", pd.Series(dtype=float)), errors="coerce").max()),
        "max_anomaly_score": float(frame_table.anomaly_score.max()),
        "min_anomaly_score": float(frame_table.anomaly_score.min()),
        "max_score_location": {"frame_index": int(max_loc_row["frame_index"]), "timestamp_sec": float(max_loc_row["timestamp_sec"])},
        "min_score_location": {"frame_index": int(min_loc_row["frame_index"]), "timestamp_sec": float(min_loc_row["timestamp_sec"])},
        "ppo_policy_mode": getattr(active_engine, "ppo_mode", "disabled") if rl_enabled else "disabled",
        "ppo_checkpoint": getattr(active_engine, "ppo_model_filename", "") if rl_enabled else "",
        "video_mime_type": video_mime_type,
        "browser_playback_ready": bool(playback_warning is None),
        "research_caveat": "Visual detector scores are normal-reference percentiles, not defect probabilities. PPO V4 and the Temporal Transformer were trained/evaluated on synthetic sequences derived from MVTec still images rather than continuous production video; RL actions are experimental triage outputs.",
    }
    dashboard_bytes = build_dashboard_html(report, frame_table)
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    results_zip = workdir / "sentinel_inspection_results.zip"
    with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(processed_video_path, arcname=processed_video_path.name)
        zf.writestr("frame_level_report.csv", csv_bytes)
        zf.writestr("summary.json", report_bytes)
        zf.writestr("inspection_dashboard.html", dashboard_bytes)
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
        "video_mime_type": video_mime_type,
        "playback_warning": playback_warning,
        "dashboard_bytes": dashboard_bytes,
    }


if uploaded_video is not None:
    st.video(uploaded_video.getvalue())
    st.caption("Uploaded source video preview. The annotated result is transcoded to browser-compatible H.264 MP4 when FFmpeg is available, so it should play inline rather than requiring a download.")
    preview_file = None
    preview_cap = None
    try:
        preview_file = tempfile.NamedTemporaryFile(suffix=Path(uploaded_video.name).suffix or ".mp4", delete=False)
        preview_file.write(uploaded_video.getvalue())
        preview_file.close()
        preview_cap = cv2.VideoCapture(preview_file.name)
        ok_preview, first_preview = preview_cap.read()
        if ok_preview and first_preview is not None:
            preview_bbox = compute_roi_bbox(
                first_preview.shape[:2], roi_mode,
                x_pct=roi_x_pct, y_pct=roi_y_pct,
                w_pct=roi_w_pct, h_pct=roi_h_pct,
            )
            px1, py1, px2, py2 = preview_bbox
            roi_preview = first_preview[py1:py2, px1:px2]
            st.image(
                cv2.cvtColor(resize_for_display(roi_preview, max_w=640, max_h=420), cv2.COLOR_BGR2RGB),
                caption=f"Preview of model input region · {roi_mode} · source ROI {(px1, py1, px2, py2)}",
                use_container_width=True,
            )
            if roi_mode == "Full frame" and abs(first_preview.shape[1] / max(first_preview.shape[0], 1) - 1.0) > 0.15:
                st.warning(
                    "Full frame is not square, but the MVTec detectors were trained on square images and resize inputs to square. "
                    "This can distort the object and increase false positives; prefer Center square crop or Custom ROI."
                )
    except Exception as preview_exc:
        st.caption(f"ROI preview unavailable: {preview_exc}")
    finally:
        if preview_cap is not None:
            preview_cap.release()
        if preview_file is not None:
            try:
                os.unlink(preview_file.name)
            except OSError:
                pass

    video_mb = len(uploaded_video.getbuffer()) / (1024 ** 2)
    info_cols = st.columns(5)
    info_cols[0].metric("Video file", uploaded_video.name[:22])
    info_cols[1].metric("Upload size", f"{video_mb:.1f} MB")
    info_cols[2].metric("Mode", "RL-assisted" if enable_rl else "Detector only")
    info_cols[3].metric("Sampling", f"every {sampling_interval:g}s")
    info_cols[4].metric("Frame budget", str(max_frames))
    run_clicked = st.button("🚀  Inspect this video", type="primary", use_container_width=True, disabled=engine is None)
    if run_clicked:
        with st.spinner("Running selected inspection route…"):
            try:
                result = process_video_file(
                    uploaded_video.getvalue(), uploaded_video.name, engine, category,
                    sampling_interval, max_frames, enable_subtypes,
                    roi_mode=roi_mode,
                    roi_x_pct=roi_x_pct, roi_y_pct=roi_y_pct,
                    roi_w_pct=roi_w_pct, roi_h_pct=roi_h_pct,
                    inspection_mode=inspection_mode,
                    decision_boundary=decision_boundary,
                )
                st.session_state["sentinel_result"] = result
                st.session_state["sentinel_category"] = category
                st.session_state["sentinel_dashboard_requested"] = False
            except Exception as exc:
                st.error("Video inspection failed. The exception below identifies the failing component.")
                st.exception(exc)

# Results persist across Streamlit reruns caused by downloads and dashboard controls.
result = st.session_state.get("sentinel_result")
if result is not None:
    st.markdown("---")
    st.markdown("<div class='section-kicker'>03 / Inspection results</div>", unsafe_allow_html=True)
    rpt = result["report"]
    table = result["table"]
    rl_mode_result = rpt.get("inspection_mode") == "RL-assisted inspection"
    actions = rpt.get("ppo_action_counts", {}) or {}

    st.markdown(
        f"<div class='panel'><b>{html_lib.escape(str(rpt.get('video_name', 'Inspection video')))}</b> · "
        f"{html_lib.escape(str(rpt.get('category', '')).title())} · "
        f"{html_lib.escape(str(rpt.get('inspection_mode', '')))}<div class='muted' style='margin-top:6px'>"
        f"Detector: {html_lib.escape(str(rpt.get('detector', '')))} · ROI: {html_lib.escape(str(rpt.get('roi_mode', '')))} · "
        f"Sampled cadence: {float(rpt.get('effective_sampling_interval_seconds', 0)):.3f}s</div></div>",
        unsafe_allow_html=True,
    )
    if result.get("playback_warning"):
        st.warning(result["playback_warning"])

    if rpt.get("high_anomaly_frame_count", 0) >= max(1, int(0.70 * rpt.get("sampled_frames", 0))):
        st.warning(
            f"At least 70% of sampled frames were labelled ANOMALY under this category’s decision policy (score ≥ {decision_boundary:.3f}). "
            "This can indicate unusual footage, but may also indicate domain shift, incorrect category, lighting differences, or ROI/background mismatch. "
            "The score is a normal-reference percentile, not a probability of defect."
        )

    metric_cols = st.columns(6 if rl_mode_result else 5)
    metric_cols[0].metric("Frames analysed", f"{rpt['sampled_frames']:,}")
    metric_cols[1].metric("Predicted anomalies", f"{rpt['high_anomaly_frame_count']:,}", help=f"Category decision policy: calibrated anomaly score ≥ {decision_boundary:.3f}.")
    metric_cols[2].metric("Mean score", f"{rpt['mean_anomaly_score']:.3f}")
    metric_cols[3].metric("Peak score", f"{rpt['max_anomaly_score']:.3f}", help=f"Frame {rpt['max_score_location']['frame_index']} at {rpt['max_score_location']['timestamp_sec']:.2f}s")
    metric_cols[4].metric("Lowest score", f"{rpt['min_anomaly_score']:.3f}", help=f"Frame {rpt['min_score_location']['frame_index']} at {rpt['min_score_location']['timestamp_sec']:.2f}s")
    if rl_mode_result:
        metric_cols[5].metric("Human-review events", f"{rpt.get('human_review_event_count', 0):,}", help=f"{rpt.get('review_action_frames', 0)} sampled frame(s) were labelled HUMAN_REVIEW")

    # Score timeline: fixed [0, 1] axis prevents Streamlit auto-scaling from making
    # score/cutoff lines look coincident or misleading. The raw/pre-final score chart
    # below makes ECDF saturation visible without changing the calibrated scores.
    timeline = table.copy()
    timeline["timestamp_sec"] = pd.to_numeric(timeline["timestamp_sec"], errors="coerce")
    timeline["frame_index"] = pd.to_numeric(timeline["frame_index"], errors="coerce")
    timeline["anomaly_score"] = pd.to_numeric(timeline["anomaly_score"], errors="coerce")
    timeline = timeline.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp_sec", "anomaly_score"])
    timeline = timeline.sort_values(["timestamp_sec", "frame_index"], kind="stable")
    st.markdown("### Calibrated anomaly-score timeline")
    st.caption("Fixed vertical axis: 0.00–1.00. The dashed amber line is the active category-specific decision cutoff; the score line is not a defect probability.")

    series_parts = [timeline[["frame_index", "timestamp_sec", "anomaly_score"]].rename(columns={"anomaly_score": "value"}).assign(series="Calibrated anomaly score")]
    series_names = ["Calibrated anomaly score"]
    series_colors = ["#2dd4bf"]
    if rl_mode_result and "temporal_anomaly_probability" in timeline.columns:
        temporal_values = pd.to_numeric(timeline["temporal_anomaly_probability"], errors="coerce")
        if temporal_values.notna().any():
            series_parts.append(timeline[["frame_index", "timestamp_sec"]].assign(value=temporal_values).dropna(subset=["value"]).assign(series="Temporal Transformer context"))
            series_names.append("Temporal Transformer context")
            series_colors.append("#a78bfa")
    plot_data = pd.concat(series_parts, ignore_index=True)
    line_chart = alt.Chart(plot_data).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("timestamp_sec:Q", title="Video timestamp (seconds)", axis=alt.Axis(tickCount=8, format=".1f")),
        y=alt.Y("value:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False), axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1.0], format=".2f")),
        color=alt.Color("series:N", title="Series", scale=alt.Scale(domain=series_names, range=series_colors)),
        detail=alt.Detail("series:N"),
        order=alt.Order("timestamp_sec:Q"),
        tooltip=[alt.Tooltip("frame_index:Q", title="Source frame", format=".0f"), alt.Tooltip("timestamp_sec:Q", title="Time (s)", format=".3f"), alt.Tooltip("series:N", title="Series"), alt.Tooltip("value:Q", title="Score", format=".5f")],
    )
    threshold_rule = alt.Chart(pd.DataFrame({"threshold": [float(decision_boundary)]})).mark_rule(
        color="#f59e0b", strokeDash=[7, 4], strokeWidth=2.5
    ).encode(y=alt.Y("threshold:Q", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False)))
    score_chart = alt.layer(line_chart, threshold_rule).resolve_scale(y="shared").properties(height=280).interactive()
    st.altair_chart(score_chart, use_container_width=True)

    score_values = timeline["anomaly_score"].astype(float).to_numpy()
    saturated_mask = score_values >= (1.0 - 1e-9)
    saturated_count = int(saturated_mask.sum())
    saturation_rate = float(saturated_mask.mean()) if len(saturated_mask) else 0.0
    unique_count = int(pd.Series(score_values).round(8).nunique()) if len(score_values) else 0
    diag1, diag2, diag3, diag4 = st.columns(4)
    diag1.metric("Minimum calibrated score", f"{float(np.min(score_values)):.4f}" if len(score_values) else "—")
    diag2.metric("Maximum calibrated score", f"{float(np.max(score_values)):.4f}" if len(score_values) else "—")
    diag3.metric("Unique score values", f"{unique_count:,}")
    diag4.metric("Scores at 1.000", f"{saturated_count:,} ({saturation_rate:.1%})")
    if saturated_count:
        st.info(
            f"{saturated_count} of {len(score_values)} sampled frame(s) ({saturation_rate:.1%}) scored exactly 1.000. "
            "This can be genuine ECDF saturation: any pre-calibration score at or above the maximum of the saved normal-reference scores maps to 1.000. "
            "The graph is fixed to 0–1, so a flat line at the top is not hidden by axis auto-scaling. The second graph shows the pre-final-calibration score for diagnosis."
        )
    if "pre_final_calibration_score" in timeline.columns:
        raw_plot = timeline[["frame_index", "timestamp_sec", "pre_final_calibration_score"]].copy()
        raw_plot["pre_final_calibration_score"] = pd.to_numeric(raw_plot["pre_final_calibration_score"], errors="coerce")
        raw_plot = raw_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["pre_final_calibration_score"])
        if not raw_plot.empty:
            raw_values = raw_plot["pre_final_calibration_score"].to_numpy(dtype=float)
            raw_min, raw_max = float(np.min(raw_values)), float(np.max(raw_values))
            raw_span = raw_max - raw_min
            raw_scale = max(abs(raw_min), abs(raw_max), raw_span, 1e-8)
            raw_pad = max(raw_span * 0.10, raw_scale * 0.02, 1e-8)
            st.markdown("### Score before final ECDF calibration")
            st.caption("Diagnostic view with its own automatically padded y-axis. For grid this is the STFPM residual score; for the other categories it is the combined YOLO/PatchCore percentile score that enters the final ECDF. Compare it within the selected category, not across detector types.")
            raw_chart = alt.Chart(raw_plot).mark_line(point=True, strokeWidth=2.2, color="#60a5fa").encode(
                x=alt.X("timestamp_sec:Q", title="Video timestamp (seconds)", axis=alt.Axis(tickCount=8, format=".1f")),
                y=alt.Y("pre_final_calibration_score:Q", title="Pre-final-calibration score", scale=alt.Scale(domain=[raw_min - raw_pad, raw_max + raw_pad], nice=False, zero=False)),
                tooltip=[alt.Tooltip("frame_index:Q", title="Source frame", format=".0f"), alt.Tooltip("timestamp_sec:Q", title="Time (s)", format=".3f"), alt.Tooltip("pre_final_calibration_score:Q", title="Pre-final-calibration score", format=".8g")],
            ).properties(height=220).interactive()
            st.altair_chart(raw_chart, use_container_width=True)
    threshold_source = next((str(r.get("decision_threshold_source", "")) for r in result.get("rows", []) if r.get("decision_threshold_source")), "category thresholds")
    st.caption(f"Active category: {category}. Decision cutoff: {decision_boundary:.4f}. Threshold source: {threshold_source}. Calibrated score = empirical percentile against normal-reference images, not a defect probability. PPO V4's internal temporal 0.95 reference is separate from this detector cutoff.")

    dashboard_col, hide_col, export_col = st.columns([1, 1, 2])
    with dashboard_col:
        if st.button("✨ Create dashboard", type="primary", use_container_width=True, key="create_dashboard"):
            st.session_state["sentinel_dashboard_requested"] = True
    with hide_col:
        if st.button("Hide dashboard", use_container_width=True, key="hide_dashboard"):
            st.session_state["sentinel_dashboard_requested"] = False
    with export_col:
        st.download_button(
            "⬇ Export complete results pack",
            data=result["zip_path"].read_bytes(),
            file_name="sentinel_inspection_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if st.session_state.get("sentinel_dashboard_requested", False):
        st.markdown("## 📊 Inspection dashboard")
        st.caption("Generated from this run's sampled frames. Use frame number and timestamp to locate any event in the source video.")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Predicted-anomaly rate", f"{100 * rpt['high_anomaly_frame_count'] / max(rpt['sampled_frames'], 1):.1f}%")
        d2.metric("Score spread", f"{rpt['max_anomaly_score'] - rpt['min_anomaly_score']:.3f}")
        d3.metric("Score median", f"{rpt['median_anomaly_score']:.3f}")
        d4.metric("Score std. dev.", f"{rpt['std_anomaly_score']:.3f}")
        dashboard_left, dashboard_right = st.columns([1.45, 1], gap="large")
        with dashboard_left:
            st.markdown("### Highest anomaly-score frames")
            top_cols = [c for c in ["frame_index", "timestamp_sec", "sample_index", "anomaly_score", "detector_prediction", "is_anomaly", "ppo_final_action", "human_review_event"] if c in table.columns]
            top_anomalies = table.sort_values("anomaly_score", ascending=False).head(12)
            st.dataframe(top_anomalies[top_cols], use_container_width=True, hide_index=True)
        with dashboard_right:
            st.markdown("### Score distribution")
            score_hist = pd.cut(
                table["anomaly_score"].astype(float),
                bins=[-0.001, 0.50, 0.70, 0.85, 0.95, 1.001],
                labels=["0–.50", ".50–.70", ".70–.85", ".85–.95", ".95–1.00"],
                include_lowest=True,
            ).value_counts(sort=False).rename_axis("score_band").to_frame("frames")
            st.bar_chart(score_hist)
        if rl_mode_result:
            action_left, action_right = st.columns([1, 1.4], gap="large")
            with action_left:
                st.markdown("### Action mix")
                action_df = pd.Series(actions, name="sampled frames").rename_axis("action").to_frame()
                if not action_df.empty:
                    st.bar_chart(action_df)
                st.caption(f"Controller: {rpt.get('ppo_checkpoint') or rpt.get('ppo_policy_mode')}. Human-review events are deduplicated using the policy cooldown; action-frame counts may be higher.")
            with action_right:
                st.markdown("### Action locations")
                action_cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "ppo_raw_action", "ppo_final_action", "human_review_event", "review_cooldown_active_before_action"] if c in table.columns]
                action_rows = table[table["ppo_final_action"].astype(str).isin(["FLAG_ANOMALY", "HUMAN_REVIEW"])] if "ppo_final_action" in table.columns else table.head(0)
                st.dataframe(action_rows[action_cols], use_container_width=True, hide_index=True)
            review_rows = table[table["ppo_final_action"].astype(str).eq("HUMAN_REVIEW")] if "ppo_final_action" in table.columns else table.head(0)
            st.markdown("### Human-review queue")
            review_cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "score_delta", "human_review_event", "previous_3s_mean_score", "persistent_high_5s_signal", "sharp_rise_signal"] if c in review_rows.columns]
            if len(review_rows):
                st.dataframe(review_rows[review_cols], use_container_width=True, hide_index=True)
            else:
                st.info("This replay contains no HUMAN_REVIEW action. The app does not fabricate a review event.")
        st.download_button(
            "⬇ Download standalone HTML dashboard",
            data=result["dashboard_bytes"],
            file_name="sentinel_inspection_dashboard.html",
            mime="text/html",
            use_container_width=True,
        )

    tab_names = ["🎞 Annotated video", "🧩 Frame explorer", "🧾 Frame report"]
    if rl_mode_result:
        tab_names.append("🧠 RL decisions")
    tab_names.append("ℹ️ Interpretation")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.video(result["video_path"].read_bytes(), format=result.get("video_mime_type", "video/mp4"))
        st.download_button(
            "⬇ Download annotated video",
            data=result["video_path"].read_bytes(),
            file_name=result["video_path"].name,
            mime=result.get("video_mime_type", "video/mp4"),
            use_container_width=True,
        )
        st.caption("Annotated inspection summary video: the original sequence is sampled at the selected cadence, not played frame-for-frame at native FPS.")
    with tabs[1]:
        frame_i = st.slider("Choose analysed frame", 0, max(0, len(table) - 1), 0, key="frame_explorer_idx")
        row = result["rows"][frame_i]
        frame_path = Path(row.get("source_frame_path", ""))
        heat_path = Path(row.get("heat_overlay_path", ""))
        if frame_path.exists() and heat_path.exists():
            a, b = st.columns(2)
            a.image(cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB), caption=f"Source frame · frame {row.get('frame_index')} · t={float(row.get('timestamp_sec', 0)):.2f}s", use_container_width=True)
            b.image(cv2.cvtColor(cv2.imread(str(heat_path)), cv2.COLOR_BGR2RGB), caption="Anomaly heatmap overlay · existing ROI mapping preserved", use_container_width=True)
        else:
            st.info("Frame previews are not available in this work folder. Use the annotated video or download the frame report.")
        visible = table.iloc[frame_i]
        st.json({k: (None if pd.isna(v) else v) for k, v in visible.to_dict().items() if k not in ["stfpm_or_patch_details"]})
    with tabs[2]:
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download frame-level CSV",
            data=result["csv_bytes"],
            file_name="sentinel_frame_level_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    next_tab_index = 3
    if rl_mode_result:
        with tabs[next_tab_index]:
            st.markdown("### PPO action trace")
            decision_cols = [c for c in [
                "timestamp_sec", "frame_index", "anomaly_score", "score_delta", "ppo_raw_action", "ppo_final_action",
                "human_review_event", "review_cooldown_active_before_action", "sharp_rise_signal",
                "persistent_high_5s_signal", "escalation_candidate_signal", "ppo_decision_source",
            ] if c in table.columns]
            st.dataframe(table[decision_cols], use_container_width=True, hide_index=True)
            st.caption("V4 actions come from the trained 20-feature PPO policy without a rule-based inference override. Review-event count is cooldown-aware.")
        next_tab_index += 1
    with tabs[next_tab_index]:
        st.markdown(
            """
            **How to read the output**

            - **Anomaly score:** empirical percentile relative to saved normal calibration images. Higher means more unusual; it is not a defect probability.
            - **Heatmap:** relative local model evidence. Color is normalized per frame for display; compare numeric scores, not colors across different frames.
            - **ROI:** the existing ROI pipeline scores only the selected region and maps the heatmap back into that region. This version preserves that behavior.
            - **Detection-only mode:** runs the visual detector and heatmap without loading or executing Transformer/PPO components.
            - **RL-assisted mode:** shows PPO actions `MONITOR`, `FLAG_ANOMALY`, or `HUMAN_REVIEW`. The standalone PPO V4's observation does not include the Transformer probability.
            - **Decision thresholds:** the five category-specific thresholds selected by F2 on `controller_dev` are bundled with the app. A project-local `assets/decision_thresholds.json` overrides them when present. The continuous score is a calibrated normal-reference percentile, not a defect probability.
            - **Score saturation:** the empirical CDF maps values at or above the maximum normal-reference score to 1.000. The app reports how many frames saturate and plots the pre-final-calibration score separately for diagnosis.

            **Research limitation:** the Temporal Transformer and PPO were trained on synthetic sequences derived from static MVTec AD still images, not continuous production footage. RL decisions on arbitrary videos remain experimental; confirm important findings with a human inspector.
            """
        )
else:
    st.markdown("---")
    st.markdown("<div class='section-kicker'>How it works</div>", unsafe_allow_html=True)
    step1, step2, step3, step4 = st.columns(4)
    step1.markdown("<div class='panel'><b>01 · Detect</b><br><span class='muted'>STFPM for grid; YOLO26 + PatchCore-style local memory for zipper, hazelnut, leather and toothbrush.</span></div>", unsafe_allow_html=True)
    step2.markdown("<div class='panel'><b>02 · Localize</b><br><span class='muted'>Calibrated anomaly score and ROI-aligned visual heatmap.</span></div>", unsafe_allow_html=True)
    step3.markdown("<div class='panel'><b>03 · Compare</b><br><span class='muted'>Track the model prediction, continuous score, maximum/minimum frame and timestamp.</span></div>", unsafe_allow_html=True)
    step4.markdown("<div class='panel'><b>04 · Triage (optional)</b><br><span class='muted'>Enable RL-assisted mode to add recall-balanced PPO V4 decisions and a human-review queue.</span></div>", unsafe_allow_html=True)

st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#73849b;font-size:.78rem'><span>SENTINEL · HYBRID VISUAL INSPECTION</span><span>Local inference · ROI-aligned heatmaps · Frame-traceable review</span></div>", unsafe_allow_html=True)
