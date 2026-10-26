from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from team_inference import (
    CATEGORIES, DEFAULT_THRESHOLD_PERCENTILE, METHODS, METHOD_LABELS,
    embed_rgb_frames, load_category_reference, score_embeddings,
    load_q_policy, choose_q_action, describe_action, expected_policy_path,
    expected_scores_path,
)

st.set_page_config(page_title="Adaptive Visual Inspection", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
      .stApp { background: radial-gradient(ellipse at 12% 0%, rgba(22,163,140,.10), transparent 34%), #0b1020; color: #edf3ff; }
      [data-testid="stSidebar"] { background: #10182a; border-right: 1px solid rgba(148,163,184,.16); }
      .hero { border: 1px solid rgba(94,234,212,.23); border-radius: 22px; padding: 24px 28px; background: linear-gradient(135deg, rgba(15,39,53,.95), rgba(18,27,47,.98) 58%, rgba(42,26,59,.92)); margin-bottom: 18px; }
      .eyebrow { font-family:'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; color:#5eead4; font-size:.74rem; }
      .hero h1 { color:#f8fafc; font-size:clamp(2rem,4vw,3rem); line-height:1.08; margin:9px 0 12px; font-weight:800; }
      .hero p { color:#bac8db; max-width:900px; }
      .panel { border:1px solid rgba(148,163,184,.18); border-radius:15px; background:rgba(17,26,45,.78); padding:16px 18px; }
      .muted { color:#9fb0c7; font-size:.86rem; }
      div.stButton > button[kind="primary"] { border:0; border-radius:12px; background:linear-gradient(100deg,#0ea5a4,#4f46e5); color:white; font-weight:800; min-height:2.8rem; }
      hr { border-color:rgba(148,163,184,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Route 02 · adaptive visual inspection</div>
      <h1>ResNet‑18 anomaly detection<br>+ inspection scheduling</h1>
      <p>Explore the three vision methods from the project notebook—kNN cosine, Mahalanobis distance, and One-Class SVM—using the same ResNet‑18 embeddings. The original Step 2 Q-learning scheduler is available with Mahalanobis scores, its intended input. Every method shows its fixed normal-reference threshold and an inspection dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.page_link("app3.py", label="← Back to Project Hub", icon="🏠")

with st.sidebar:
    st.markdown("## 🧭 Inspection setup")
    default_data = os.environ.get("TEAM_MVTEC_ROOT", str(ROOT / "data" / "mvtec"))
    data_root = Path(default_data).expanduser()
    exported_reference_dir = ROOT / "team_artifacts" / "vision"
    exported_refs_available = all((exported_reference_dir / f"reference_{cat}.npz").is_file() for cat in CATEGORIES)
    if exported_refs_available:
        st.success("Colab reference assets found · local MVTec dataset is not required")
    else:
        st.warning("Colab reference assets are incomplete. The fallback requires local MVTec train/good images.")
        data_root = Path(st.text_input(
            "Local MVTec data root",
            value=str(data_root),
            help="Folder directly containing grid/, zipper/, hazelnut/, leather/, and toothbrush/.",
        )).expanduser()

    device = st.selectbox("Compute device", ["auto", "cuda", "cpu"], index=0)
    category = st.selectbox("Object category", CATEGORIES, index=0)
    method_options = ["kNN cosine (k=5)", "Mahalanobis", "One-Class SVM"]
    method_label = st.selectbox("Vision method", method_options, index=1)
    method = {
        "kNN cosine (k=5)": "knn_cosine",
        "Mahalanobis": "mahalanobis",
        "One-Class SVM": "ocsvm",
    }[method_label]
    if method == "knn_cosine":
        st.caption("Mean cosine distance to the 5 nearest normal training embeddings.")
    elif method == "mahalanobis":
        st.caption("Ledoit–Wolf-shrunk Mahalanobis distance from the normal embedding distribution. This is the Step 2 Q-learning input.")
    else:
        st.caption("RBF One-Class SVM anomaly score (negative decision function), trained on normal embeddings.")
    sampling_interval = st.select_slider(
        "Sample one frame every",
        options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        value=0.5,
        format_func=lambda v: f"{v:g} sec",
    )
    max_frames = st.slider("Maximum sampled frames", min_value=8, max_value=240, value=72, step=8)

    # Show the fixed category-specific cutoff. It is derived from the 95th percentile
    # of cross-validated scores for normal train/good embeddings, not selected by user.
    threshold_info = None
    reference_error = None
    try:
        reference_preview = load_category_reference(
            data_root, category, exported_reference_dir, device
        )
        normal_scores = np.sort(np.asarray(reference_preview.calibration_scores[method], dtype=np.float64))
        if normal_scores.size == 0:
            raise ValueError("The kNN normal-reference calibration scores are empty.")
        fixed_raw_threshold = float(np.percentile(normal_scores, DEFAULT_THRESHOLD_PERCENTILE))
        fixed_normalized_threshold = float(np.clip(
            np.searchsorted(normal_scores, fixed_raw_threshold, side="left") / len(normal_scores),
            0.0, 1.0,
        ))
        threshold_info = (fixed_raw_threshold, fixed_normalized_threshold, len(normal_scores))
    except Exception as exc:
        reference_error = str(exc)

    st.markdown("---")
    st.markdown("### 🔒 Fixed anomaly threshold")
    if threshold_info:
        raw_cutoff, normalized_cutoff, n_reference = threshold_info
        st.metric("Normalized cutoff", f"{normalized_cutoff:.3f}")
        t1, t2 = st.columns(2)
        t1.metric("Calibration percentile", f"{DEFAULT_THRESHOLD_PERCENTILE}th")
        t2.metric("Normal reference scores", f"{n_reference:,}")
        raw_score_name = {
            "knn_cosine": "raw cosine-distance cutoff",
            "mahalanobis": "raw Mahalanobis-distance cutoff",
            "ocsvm": "raw One-Class SVM score cutoff",
        }[method]
        st.caption(f"Fixed {raw_score_name}: `{raw_cutoff:.6f}`. Derived from cross-validated normal `train/good` scores; it is visible for transparency and cannot be edited here.")
    else:
        st.warning("Threshold unavailable until the reference assets load.")
        st.caption(reference_error or "Check the exported references.")

    policy_path = expected_policy_path(ROOT)
    policy_file_exists = policy_path.is_file()
    policy_error = None
    policy = {}
    if method == "mahalanobis" and policy_file_exists:
        try:
            policy = load_q_policy(policy_path, expected_method="mahalanobis")
            if not policy:
                raise ValueError("The Q-policy file contains no usable category/risk states.")
        except Exception as exc:
            policy_error = str(exc)
    policy_available = method == "mahalanobis" and policy_file_exists and bool(policy) and policy_error is None
    if method != "mahalanobis":
        st.info("The supplied Step 2 scheduler was trained on Mahalanobis outputs. It is intentionally not applied to this detector; switch to Mahalanobis to enable Q-learning.")
    elif policy_available:
        st.success("Step 2 Mahalanobis Q-learning policy found")
    elif policy_error:
        st.error(policy_error)
    else:
        st.warning("Mahalanobis Q-learning policy not found. Run `python rebuild_mahalanobis_policy.py` if the exported Step 1 score CSV is available, or regenerate the Colab asset bundle.")

    use_rl = st.checkbox(
        "Enable Q-learning inspection scheduler (Mahalanobis only)",
        value=policy_available,
        disabled=not policy_available,
        help="Matches the original Step 2 notebook: category + risk-level states, actions skip/prediction/expert, and the original reward costs/penalties.",
    )
    st.markdown("---")
    st.caption("All route outputs stay under team_artifacts/. Sentinel's models/ and assets/ remain separate.")

upload_col, route_col = st.columns([1.25, 0.75], gap="large")
with upload_col:
    st.markdown("### 01 / Upload video")
    uploaded_video = st.file_uploader("Upload an inspection video", type=["mp4", "mov", "avi", "mkv", "mpeg", "mpg"], key="team_video_upload")
with route_col:
    st.markdown("### 02 / Active route")
    st.markdown(
        f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()}</div><div style='color:#5eead4;margin-top:6px;font-weight:700'>{method_label}</div><div class='muted' style='margin-top:9px'>ResNet‑18 512D embedding → fixed calibrated threshold → " + ("Step 2 Q-learning action" if use_rl else "normal/anomaly label") + "</div></div>",
        unsafe_allow_html=True,
    )
    st.caption("Frame-level anomaly scoring only; this approach does not generate pixel-level defect heatmaps.")

active_rl = bool(use_rl and policy_available)


def sample_video(uploaded, interval_sec: float, limit: int):
    suffix = Path(uploaded.name).suffix or ".mp4"
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp_video.write(uploaded.getbuffer())
        temp_video.close()
        cap = cv2.VideoCapture(temp_video.name)
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open this video. Try an MP4/MOV file encoded with a standard codec.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            raise RuntimeError("Could not read the video frame count.")
        stride = max(1, int(round(fps * float(interval_sec))))
        candidate_indices = np.arange(0, total_frames, stride, dtype=np.int64)
        if len(candidate_indices) > limit:
            positions = np.linspace(0, len(candidate_indices) - 1, num=limit, dtype=np.int64)
            candidate_indices = candidate_indices[positions]
        frames_bgr = []
        timestamps = []
        frame_indices = []
        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames_bgr.append(frame)
            frame_indices.append(int(idx))
            timestamps.append(float(idx) / fps)
        cap.release()
        if not frames_bgr:
            raise RuntimeError("No frames could be sampled from this video.")
        return temp_video.name, fps, total_frames, frame_indices, timestamps, frames_bgr
    except Exception:
        try:
            os.unlink(temp_video.name)
        except OSError:
            pass
        raise


def draw_label(frame: np.ndarray, row: dict) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    label = f"{row['prediction_label']}  score={row['anomaly_score']:.3f}"
    if row.get("risk_level"):
        label += f"  risk={row['risk_level']}"
    action = row.get("scheduler_action")
    if action:
        label += f"  RL={action.upper()}"
    cv2.rectangle(out, (0, 0), (width, min(64, height)), (14, 22, 38), -1)
    cv2.putText(out, label[:110], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 240, 215) if row["prediction"] else (150, 225, 165), 2, cv2.LINE_AA)
    cv2.putText(out, f"t={row['timestamp_sec']:.2f}s | frame={row['frame_index']}", (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 241), 1, cv2.LINE_AA)
    return out


def write_sampled_clip(frames, interval, output_path: Path):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    # Output is a review montage/clip of sampled frames, not a frame-for-frame real-time rendering.
    output_fps = float(np.clip(1.0 / max(float(interval), 0.1), 1.0, 12.0))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (w, h))
    if not writer.isOpened():
        return None
    for frame in frames:
        writer.write(frame)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg:
        h264_path = output_path.with_name(output_path.stem + "_h264.mp4")
        try:
            result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(output_path), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(h264_path)], capture_output=True, text=True, timeout=180)
            if result.returncode == 0 and h264_path.is_file() and h264_path.stat().st_size > 1000:
                return h264_path
        except Exception:
            pass
    return output_path


if uploaded_video is None:
    st.info("Upload a short clip to start. The app uses the exported Colab reference embeddings when available; the local MVTec dataset is not needed when the reference bundle is complete.")
else:
    st.video(uploaded_video)
    start = st.button("Run Inspection", type="primary", use_container_width=True)
    if start:
        status = st.status("Preparing adaptive inspection…", expanded=True)
        temp_path = None
        try:
            status.update(label=f"Loading {method_label} normal-reference calibration…", state="running")
            reference = load_category_reference(data_root, category, ROOT / "team_artifacts" / "vision", device)
            if not active_rl and use_rl:
                status.write("Q-learning is unavailable for this selection; continuing with the selected vision detector and fixed threshold.")
            status.write(f"Reference features ready: {len(reference.train_features):,} normal training images · fingerprint `{reference.fingerprint}`")
            status.update(label="Sampling video frames…", state="running")
            temp_path, source_fps, total_frames, frame_indices, timestamps, frames_bgr = sample_video(uploaded_video, sampling_interval, max_frames)
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
            status.write(f"Sampled {len(frames_rgb)} frames from {total_frames:,} source frames at {source_fps:.2f} FPS.")
            status.update(label=f"Extracting embeddings and scoring with {method_label}…", state="running")
            embeddings = embed_rgb_frames(frames_rgb, device=device)
            scores = score_embeddings(reference, embeddings, method)
            result_rows = []
            annotated = []
            for idx, ts, frame, score in zip(frame_indices, timestamps, frames_bgr, scores):
                action = choose_q_action(policy, category, score["risk_level"]) if active_rl else None
                next_step = describe_action(action) if active_rl else ("Flag for review" if score["prediction"] else "No anomaly flagged by threshold")
                if action == "expert":
                    final_decision = "PENDING EXPERT REVIEW"
                elif action == "skip":
                    final_decision = "SKIP FURTHER INSPECTION"
                else:
                    final_decision = "ANOMALY" if score["prediction"] else "NORMAL"
                row = {
                    "frame_index": idx,
                    "timestamp_sec": ts,
                    "category": category,
                    "method": method,
                    "raw_score": score["raw_score"],
                    "raw_threshold": score["threshold"],
                    "anomaly_score": score["normalized_score"],
                    "decision_threshold": score["normalized_threshold"],
                    "prediction": score["prediction"],
                    "prediction_label": score["prediction_label"],
                    "risk_level": score["risk_level"],
                    "scheduler_action": action or "not_used",
                    "recommended_next_step": next_step,
                    "final_decision": final_decision,
                }
                result_rows.append(row)
                annotated.append(draw_label(frame, row))
            results_df = pd.DataFrame(result_rows)
            out_dir = ROOT / "team_artifacts" / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(uploaded_video.name).stem.replace(" ", "_")[:48]
            csv_path = out_dir / f"{stem}_{category}_{method}_results.csv"
            results_df.to_csv(csv_path, index=False)
            clip_path = write_sampled_clip(annotated, sampling_interval, out_dir / f"{stem}_{category}_sampled_review.mp4")
            st.session_state["team_last_results"] = results_df
            st.session_state["team_last_rl"] = bool(active_rl)
            st.session_state["team_last_category"] = category
            st.session_state["team_last_method"] = method
            st.session_state["team_last_method_label"] = method_label
            st.session_state["team_last_csv"] = str(csv_path)
            st.session_state["team_last_clip"] = str(clip_path) if clip_path else ""
            st.session_state["team_last_frames"] = annotated
            status.update(label="Inspection run finished", state="complete")
        except Exception as exc:
            status.update(label="Inspection run could not finish", state="error")
            st.error(str(exc))
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

if "team_last_results" in st.session_state:
    results_df = st.session_state["team_last_results"]
    st.markdown("---")
    st.markdown("## Inspection dashboard")
    total = len(results_df)
    anomalies = int((results_df["prediction"] == 1).sum())
    last_run_rl = bool(st.session_state.get("team_last_rl", False))
    last_run_category = st.session_state.get("team_last_category", "unknown")
    last_run_method = st.session_state.get("team_last_method", "knn_cosine")
    last_run_method_label = st.session_state.get("team_last_method_label", METHOD_LABELS.get(last_run_method, last_run_method))
    reviewed = int((results_df["scheduler_action"] == "expert").sum())
    normal_count = total - anomalies
    anomaly_rate = anomalies / max(total, 1)
    mean_score = float(results_df["anomaly_score"].mean())
    shown_threshold = float(results_df["decision_threshold"].iloc[0]) if total else float("nan")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames scored", f"{total:,}")
    c2.metric("Anomalies flagged", f"{anomalies:,}")
    c3.metric("Anomaly rate", f"{anomaly_rate:.1%}")
    c4.metric("Mean normalized score", f"{mean_score:.3f}")
    c5.metric("Fixed normalized threshold", f"{shown_threshold:.3f}")

    shown_raw_threshold = float(results_df["raw_threshold"].iloc[0]) if "raw_threshold" in results_df.columns and total else float("nan")
    st.caption(f"Last run: **{last_run_category} · {last_run_method_label}**. Fixed raw cutoff: `{shown_raw_threshold:.6f}`; selected at the {DEFAULT_THRESHOLD_PERCENTILE}th percentile of cross-validated normal-reference scores. Thresholds and score scales differ by detector and should not be compared directly with Sentinel.")

    chart_col, action_col = st.columns([1.5, 1], gap="large")
    with chart_col:
        st.markdown(f"### Anomaly score timeline · {last_run_method_label}")
        chart_df = results_df[["timestamp_sec", "anomaly_score", "decision_threshold"]].copy()
        chart_df = chart_df.rename(columns={
            "timestamp_sec": "Time (seconds)",
            "anomaly_score": "Normalized anomaly score",
            "decision_threshold": "Fixed threshold",
        }).set_index("Time (seconds)")
        st.line_chart(chart_df, use_container_width=True)
        st.caption("Each point represents a sampled frame, not every source-video frame.")
    with action_col:
        st.markdown("### Frame outcomes")
        outcome_df = pd.DataFrame({
            "Outcome": ["NORMAL", "ANOMALY"],
            "Frames": [normal_count, anomalies],
        }).set_index("Outcome")
        st.bar_chart(outcome_df, use_container_width=True)
        if last_run_rl:
            st.markdown("### Q-learning actions")
            action_counts = results_df["scheduler_action"].replace("not_used", "not used").value_counts().rename_axis("Action").to_frame("Frames")
            st.bar_chart(action_counts, use_container_width=True)
            st.metric("Expert-review recommendations", f"{reviewed:,}")
        else:
            st.info("Q-learning was not used for this run; outcomes are predictions from the selected vision detector and its fixed threshold.")

    with st.expander("What can be compared across the two approaches?"):
        st.markdown(
            "Only the shared frame-level **NORMAL / ANOMALY** outcome can be used for a cautious qualitative comparison, and only when the category and sampled frames align. The underlying score scales, thresholds, heatmaps, and scheduler actions are method-specific and must not be ranked directly against Sentinel. This page does not fuse or cross-calibrate the two models."
        )

    if st.session_state.get("team_last_clip") and Path(st.session_state["team_last_clip"]).is_file():
        st.markdown("### Sampled-frame review clip")
        st.caption("This clip contains the sampled frames in order, with the original frame index and timestamp overlaid. It is not a frame-for-frame export of the source video.")
        clip_path = Path(st.session_state["team_last_clip"])
        st.video(str(clip_path))
        st.download_button("Download annotated review clip", data=clip_path.read_bytes(), file_name=clip_path.name, mime="video/mp4")

    display_cols = ["frame_index", "timestamp_sec", "raw_score", "anomaly_score", "decision_threshold", "prediction_label", "risk_level", "scheduler_action", "recommended_next_step", "final_decision"]
    display_df = results_df[display_cols].rename(columns={
        "frame_index": "Frame",
        "timestamp_sec": "Time (s)",
        "raw_score": f"Raw score ({last_run_method_label})",
        "anomaly_score": "Normalized score",
        "decision_threshold": "Fixed threshold",
        "prediction_label": "Binary outcome",
        "risk_level": "Risk level",
        "scheduler_action": "Q-learning action",
        "recommended_next_step": "Recommended next step",
        "final_decision": "Final status",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    csv_path = Path(st.session_state["team_last_csv"])
    if csv_path.is_file():
        st.download_button("Download frame-level CSV", data=csv_path.read_bytes(), file_name=csv_path.name, mime="text/csv")
    with st.expander("Method notes and limitations"):
        st.markdown(
            """
            - The vision representation follows Step 1: ImageNet-pretrained ResNet‑18 with the classification layer replaced by a 512-dimensional embedding.
            - The three Step 1 detectors are available: kNN cosine distance (`k=5`), Ledoit–Wolf Mahalanobis distance, and RBF One-Class SVM (`nu=0.1`).
            - For each selected method, the visible but locked raw cutoff is the 95th percentile of five-fold cross-validated scores from normal `train/good` images. The 0–1 normalized score is its empirical percentile against those normal-reference scores.
            - The original Step 2 scheduler consumes **Mahalanobis** scores only. It uses states `(category, risk_level)` and actions `skip`, `prediction`, and `expert`, with the notebook's illustrative action costs and missed-defect / false-rejection penalties. Q-learning is intentionally disabled for kNN and One-Class SVM rather than applying a policy trained on a different score distribution.
            - Ground-truth labels are used offline to simulate rewards and evaluate held-out records, not at video inference time. The notebook assumes expert review is perfect in simulation; the runtime action is only a recommendation for a real human.
            - MVTec still-image training data and real uploaded video can have substantial domain shift. Treat results as research output, not a validated production inspection decision.
            """
        )

st.sidebar.markdown("---")
st.sidebar.caption("All three Step 1 vision methods can run independently. The original Step 2 Q-learning scheduler is loaded only with the matching Mahalanobis method.")
