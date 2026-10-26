from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="MVTec Lab · Project Hub",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
      .stApp { background: radial-gradient(ellipse at 10% 0%, rgba(45,120,210,.15), transparent 32%), #0b1020; color: #edf3ff; }
      [data-testid="stSidebar"] { background: #10182a; border-right: 1px solid rgba(148,163,184,.16); }
      [data-testid="stHeader"] { background: rgba(11,16,32,.82); }
      .hero { border: 1px solid rgba(125,165,255,.22); border-radius: 24px; padding: 30px; background: linear-gradient(135deg, rgba(17,39,67,.96), rgba(18,27,47,.98) 58%, rgba(43,28,65,.92)); margin: 12px 0 22px; box-shadow: 0 18px 60px rgba(0,0,0,.22); }
      .eyebrow { font-family: 'DM Mono', monospace; letter-spacing: .14em; text-transform: uppercase; color: #93c5fd; font-size: .76rem; }
      .hero h1 { font-size: clamp(2.25rem,4vw,3.4rem); line-height: 1.06; margin: 10px 0 12px; color: #f8fafc; font-weight: 800; letter-spacing: -.045em; }
      .hero p { color: #c4d2e4; max-width: 900px; font-size: 1rem; }
      .chip { display:inline-block; padding: 6px 11px; border:1px solid rgba(148,163,184,.23); border-radius:99px; color:#d5deec; font-size:.78rem; margin: 4px 5px 0 0; background:rgba(15,23,42,.55); }
      .route-card { border:1px solid rgba(148,163,184,.20); border-radius:18px; background:linear-gradient(145deg,rgba(18,31,52,.94),rgba(13,21,37,.95)); padding:22px 22px 18px; min-height:260px; }
      .route-kicker { color:#93c5fd; text-transform:uppercase; letter-spacing:.12em; font-family:'DM Mono',monospace; font-size:.72rem; margin-bottom:9px; }
      .route-title { color:#f8fafc; font-size:1.25rem; font-weight:800; margin-bottom:8px; }
      .route-copy { color:#b7c5d9; font-size:.93rem; line-height:1.55; }
      .muted { color:#9fb0c7; font-size:.85rem; }
      div.stButton > button[kind="primary"] { border:0; border-radius:12px; background:linear-gradient(100deg,#2563eb,#6d28d9); color:white; font-weight:800; min-height:2.9rem; }
      div.stButton > button { border-radius:10px; font-weight:700; }
      hr { border-color:rgba(148,163,184,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Adaptive visual inspection · local lab</div>
      <h1>One workspace.<br>Two independent approaches.</h1>
      <p>Choose which inspection pipeline to run without changing browser tabs. Your Sentinel implementation keeps its existing inference code and model bundle; the independent ResNet-18 route provides the three original vision methods and uses the supplied Q-learning scheduler with its intended Mahalanobis input.</p>
      <span class="chip">VIDEO UPLOAD</span><span class="chip">FRAME SAMPLING</span><span class="chip">ANOMALY SCORING</span><span class="chip">OPTIONAL RL SCHEDULING</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Check only for the presence of code files. Model assets are validated inside each route.
my_app_ready = (ROOT / "app2.py").is_file() and (ROOT / "inference2.py").is_file()
team_page_ready = (ROOT / "team_inference.py").is_file()
team_data_default = Path(os.environ.get("TEAM_MVTEC_ROOT", str(ROOT / "data" / "mvtec"))).expanduser()
TEAM_CATEGORIES = ("grid", "zipper", "hazelnut", "leather", "toothbrush")
team_data_ready = all((team_data_default / cat / "train" / "good").is_dir() for cat in TEAM_CATEGORIES)
team_vision_assets_ready = all((ROOT / "team_artifacts" / "vision" / f"reference_{cat}.npz").is_file() for cat in TEAM_CATEGORIES)
team_encoder_ready = (ROOT / "team_artifacts" / "vision" / "resnet18_imagenet1k_v1.pth").is_file()
team_policy_path = ROOT / "team_artifacts" / "rl" / "shared_q_policy.json"

left, right = st.columns(2, gap="large")
with left:
    st.markdown(
        """
        <div class="route-card">
          <div class="route-kicker">Route 01 · Your implementation</div>
          <div class="route-title">🛡️ Sentinel — Hybrid Inspection</div>
          <div class="route-copy">Your existing YOLO26 embeddings + PatchCore-style local scoring, STFPM grid detector, calibrated thresholds, ROI controls, and optional temporal Transformer/PPO route. This opens the existing <code>app2.py</code> and continues to use <code>inference2.py</code>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if my_app_ready:
        st.success("App code found · model bundle is checked on the Sentinel page")
    else:
        st.warning("app2.py and/or inference2.py is missing from this folder")
    if st.button("Open Sentinel", type="primary", use_container_width=True, disabled=not my_app_ready):
        st.switch_page("pages/01_Sentinel.py")

with right:
    st.markdown(
        """
        <div class="route-card">
          <div class="route-kicker">Route 02 · Alternate approach</div>
          <div class="route-title">🧭 Adaptive Visual Inspection</div>
          <div class="route-copy">A separate ResNet‑18 pipeline with the notebook’s three detectors: kNN cosine, Mahalanobis, and One-Class SVM. The original Q-learning scheduler uses Mahalanobis scores to choose skip, prediction, or expert review. Includes fixed thresholds and an inspection dashboard; all assets stay in <code>team_artifacts/</code>.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if team_vision_assets_ready:
        st.success("Colab vision references found · local MVTec dataset is not required")
    elif team_data_ready:
        st.info(f"Local MVTec train/good folders found at `{team_data_default}` · App3 can build references locally")
    else:
        st.warning("kNN reference assets are not installed yet · run colab_asset_builder.ipynb and extract MVTec_Team_Assets.zip here")
    if team_policy_path.is_file():
        st.info("Q-learning policy file found · the inspection page validates the Mahalanobis method contract")
    else:
        st.caption("The scheduler is enabled when a Step 2-compatible Mahalanobis Q-policy is exported to team_artifacts/rl/shared_q_policy.json.")
    if st.button("Open Adaptive Inspection", type="primary", use_container_width=True, disabled=not team_page_ready):
        st.switch_page("pages/02_Adaptive_Inspection.py")

st.markdown("---")
st.markdown("### Comparison rule")
st.info("Compare only the shared binary NORMAL / ANOMALY outcome—and only for aligned categories and sampled frames. Numerical anomaly scores, calibration thresholds, localization heatmaps, and RL actions differ by method and are not directly comparable. The two approaches remain independent.")

st.markdown("### Separation by design")
sep1, sep2, sep3 = st.columns(3)
with sep1:
    st.markdown("**Your models stay yours**")
    st.caption("Sentinel continues to read the original `models/` and `assets/` folders.")
with sep2:
    st.markdown("**Team artifacts are separate**")
    st.caption("Reference embeddings, policy JSON, videos, and CSV results go under `team_artifacts/`.")
with sep3:
    st.markdown("**Same browser tab**")
    st.caption("Use the page selector in the left navigation or the buttons above to switch approaches.")

st.sidebar.markdown("## 🧭 MVTec Project Hub")
st.sidebar.caption("Choose a route from the main page, or use the multipage selector above this sidebar section.")
st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · video inference runs locally on the machine hosting Streamlit.")
