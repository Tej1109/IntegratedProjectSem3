# Sentinel — Hybrid MVTec Video Inspector

A polished Streamlit app for uploading an inspection video and receiving:

- frame-level anomaly scores and anomaly heatmap overlays;
- the trained category-routed detector (STFPM for `grid`, YOLO26 + PatchCore-style local k-NN for `zipper`, `hazelnut`, `leather`, and `toothbrush`);
- temporal risk estimates from the trained causal Temporal Transformer;
- PPO raw action and the same sudden-rise human-review guardrail used by the training notebook;
- optional DINOv2 + LinearSVM defect-subtype suggestions, where a saved classifier exists;
- annotated MP4, frame-level CSV, JSON summary, and a downloadable ZIP of results.

## Important — the trained weights are separate

The source-code ZIP does **not** contain your roughly 875 MB trained model directory. The app must be pointed at a folder containing the model artifacts from your own Drive project. The raw MVTec dataset is not required for inference and is not bundled.

Expected folder layout:

```text
MVTec_RL_Anomaly_Project/
├── models/
│   ├── yolo26n.pt
│   ├── wide_resnet50_2_patch_extractor.pth
│   ├── stfpm_teacher_resnet18_grid.pth
│   ├── stfpm_student_resnet18_grid.pth
│   ├── yolo26_embedding_knn_<category>.joblib
│   ├── patchcore_patch_bank_<category>.pt
│   ├── causal_temporal_transformer.pt
│   ├── temporal_transformer_input_scaler.joblib
│   ├── ppo_inspection_policy.zip
│   ├── dinov2_vits14_subtype_backbone.pth  # optional subtype feature extraction
│   └── defect_subtype_*_linearsvm.joblib   # optional
├── assets/
│   ├── calibration_<category>.npz
│   └── stfpm_calibration_grid.npz
└── model_manifest.json                    # optional
```

## A. Export just the model assets from Google Drive (Colab)

Run this cell in the Colab session where your Drive model directory exists. It zips `models/`, calibration files, and the manifest only; it deliberately excludes the MVTec dataset, raw images, masks, and feature cache.

```python
from pathlib import Path
import zipfile
from google.colab import files

PROJECT = Path('/content/drive/MyDrive/MVTec_RL_Anomaly_Project')
OUT = Path('/content/MVTEC_VIDEO_INSPECTOR_MODEL_BUNDLE.zip')
assert (PROJECT / 'models').is_dir() and (PROJECT / 'assets').is_dir()

with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
    for p in sorted((PROJECT / 'models').rglob('*')):
        if p.is_file():
            zf.write(p, p.relative_to(PROJECT))
    for p in sorted((PROJECT / 'assets').glob('*calibration*.npz')):
        if p.is_file():
            zf.write(p, p.relative_to(PROJECT))
    manifest = PROJECT / 'model_manifest.json'
    if manifest.is_file():
        zf.write(manifest, manifest.relative_to(PROJECT))

print('Bundle size (MB):', round(OUT.stat().st_size / 1e6, 1))
print('Dataset was not included.')
files.download(str(OUT))
```

Download and extract the model bundle on the machine where you want to run the app. The extracted folder should contain `models/`, `assets/`, and optionally `model_manifest.json` at its root.

## B. Install and run locally

Recommended: Python 3.11 or 3.12. For an NVIDIA GPU, install the correct CUDA-enabled PyTorch and torchvision build from [the official PyTorch selector](https://pytorch.org/get-started/locally/) first. Then install the app dependencies.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:MVTEC_PROJECT_DIR = "D:\models\MVTec_RL_Anomaly_Project"
streamlit run app.py
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export MVTEC_PROJECT_DIR="$HOME/models/MVTec_RL_Anomaly_Project"
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). The project folder can also be selected/edited in the sidebar.

## C. App usage

1. Choose the product category that matches the inspected object. Do not use a different category model just because the object looks similar.
2. Upload MP4/MOV/AVI/MKV footage.
3. Select a sampling interval and maximum frame budget. Shorter intervals produce denser inspection but take longer.
4. Click **Inspect this video**.
5. Review the annotated MP4, anomaly score timeline, frame table and PPO decisions; download the result ZIP or CSV.

The output video contains sampled frames, not every original source frame. Its playback rate approximates the sampling interval; it is an annotated inspection summary rather than a frame-perfect copy of the uploaded clip.

## Model logic and interpretation

- **Grid:** STFPM (ResNet-18 teacher/student) feature-discrepancy map. The score is an empirical percentile against saved normal calibration scores.
- **Other four categories:** YOLO26 global-embedding k-NN and Wide-ResNet50-2 local patch memory bank. The detector combines calibrated global and patch scores (0.30 / 0.70) and calibrates their ensemble against normal calibration examples.
- **Temporal Transformer:** consumes six features in training order: routed score, local evidence percentile, global evidence percentile, score delta, rolling mean over three frames, and rolling standard deviation over five frames. The app uses causal rolling windows of up to 24 sampled frames.
- **PPO:** predicts `DO_NOTHING`, `FLAG_ANOMALY`, or `HUMAN_REVIEW`. The saved guardrail overrides the action to `HUMAN_REVIEW` when anomaly score is at least 0.97 and one-step score delta is at least 0.20.
- **Subtype classification (optional):** DINOv2 features plus the saved category LinearSVM are used only where a matching classifier file exists. Treat predictions as suggestions.

An anomaly score of 0.95 means “at/above the 95th percentile of the detector's normal calibration reference,” not a 95% probability of a defect. Heatmap colors are normalized per frame for display and should not be compared numerically between videos.

## Important research caveat

The visual detectors were trained on MVTec AD images. The Temporal Transformer and PPO were trained/evaluated on synthetic sequences made from MVTec still images, not real, time-stamped production video. Running them on uploaded video is a useful integration/demo experiment, but **the resulting temporal probability and PPO decisions are not validated industrial video performance**. Use human review before taking action based on the output.

## Troubleshooting

- **Missing calibration file:** include `assets/calibration_*.npz` and `assets/stfpm_calibration_grid.npz` from Drive; weights alone are not enough to reproduce the calibrated scores.
- **sklearn/joblib version warning:** saved scikit-learn models should be loaded using a close version to the one used to train the notebook. If loading warns or fails, align the versions rather than blindly re-saving the models.
- **YOLO embedding API mismatch:** install a current `ultralytics` release supporting the YOLO26 checkpoint used by the notebook.
- **No CUDA:** choose `auto` or `cpu`; video processing will be slower.
- **Subtype model first load:** DINOv2 source code may need to be fetched from the official `facebookresearch/dinov2` repository if it is not already cached. The anomaly detection + PPO path does not require enabling this optional subtype step.
- **Large video:** lower the frame budget or use a wider sampling interval.

## Files

- `app.py` — Streamlit UI, video upload, plots, downloads.
- `inference.py` — hybrid detector, temporal transformer, PPO and optional subtype loading.
- `scripts/package_model_assets.py` — exports trained assets without the dataset.
- `requirements.txt` — Python packages.
