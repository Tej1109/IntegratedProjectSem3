# MVTec Project Hub (app3)

This add-on creates a single Streamlit entry point for the two approaches. It does **not** replace `app2.py` or `inference2.py`.

## File layout

Copy these files into the same project root that currently contains your working `app2.py` and `inference2.py`:

```text
your_project/
  app2.py                         # your existing Sentinel app; leave unchanged
  inference2.py                   # your existing Sentinel inference; leave unchanged
  app3.py                         # new landing page / project hub
  team_inference.py               # separate ResNet-18 inference pipeline
  prepare_team_assets.py          # exports the peer notebook's Q-learning policy
  pages/
    01_Sentinel.py                # runs existing app2.py as a Streamlit page
    02_Team_Approach.py           # video-upload page for the notebooks' approach
  data/
    mvtec/
      grid/train/good/...
      zipper/train/good/...
      hazelnut/train/good/...
      leather/train/good/...
      toothbrush/train/good/...
```

The team route stores its cache and outputs in `team_artifacts/`. Sentinel continues to use its own `models/` and `assets/` folders. Do not move or rename your current model files.

## Run the hub

From your existing project folder, with your normal virtual environment active:

```powershell
python -m streamlit run app3.py
```

Open the local URL Streamlit prints. Choose **Open Sentinel** or **Open Team Approach** on the hub. Streamlit multipage navigation stays in the same browser tab. The Sentinel page executes your existing `app2.py` from memory and injects only a “Project Hub” navigation link; the file on disk is not edited.

## Team visual pipeline

1. Download/extract MVTec AD so that the data root directly contains the five category folders shown above. Set `TEAM_MVTEC_ROOT` or change the root in the Team Approach page.
2. Upload a short inspection video. The first run for a category extracts ResNet-18 embeddings from its `train/good` images and creates cross-validated normal-reference scores under `team_artifacts/vision/`. Later runs reuse the cache.
3. Choose kNN cosine, Mahalanobis, or One-Class SVM. The RL scheduler is only enabled for Mahalanobis, because that is the vision method used by the Step 2 notebook.

The pretrained ResNet-18 weights may be downloaded by torchvision on first use if they are not already cached.

## Enable the Q-learning scheduler

The uploaded Step 2 notebook uses Mahalanobis scores and saves CSV outputs, but it does not export a standalone Q-table JSON. The helper script bridges that gap. You **do not have to run either notebook first** if you have the MVTec dataset locally.

From the project root, run:

```powershell
python prepare_team_assets.py --mvtec-root data\mvtec
```

This one-time setup recreates the Step 1-style `current_anomaly_score.csv` from the MVTec folder structure using ImageNet-pretrained ResNet‑18 embeddings, kNN cosine / Mahalanobis / One-Class SVM scores, and cross-validated normal-reference score calibration. It then filters the Mahalanobis rows and trains a shared Q-learning policy with the Step 2 notebook's low/medium/high risk definition, skip/prediction/expert actions, 80/20 stratified split, and illustrative reward settings. Reference features and scores are cached under `team_artifacts/`; subsequent runs reuse the feature cache. The first run may take a while and may download ResNet‑18 weights if they are not cached.

The policy is written to `team_artifacts/rl/shared_q_policy.json`. Reload the Team Approach page and enable Q-learning. During uploaded-video inference, action selection uses the category and score-derived risk level; labels are used only during the offline policy-training step. The notebook's simulated “expert” is assumed perfect during training, but the app's `expert` action is only a recommendation for a real human to review.

If you already ran the Step 1 notebook and have its CSV, you can skip recomputing vision scores:

```powershell
python prepare_team_assets.py --scores-csv "C:\path\to\current_anomaly_score.csv"
```

The helper intentionally does not train the notebook's optional defect-type classifier. The video app does not use that classifier, and its core scorer/scheduler fields are generated separately.

## Dependencies

Your existing environment likely already has Streamlit, torch, torchvision, OpenCV, NumPy, pandas, Pillow, and scikit-learn. If needed:

```powershell
python -m pip install streamlit torch torchvision opencv-python numpy pandas pillow scikit-learn
```

`imageio-ffmpeg` is optional and helps make the sampled-frame review clip browser-compatible:

```powershell
python -m pip install imageio-ffmpeg
```

## Important scope note

The Step 1 notebook uses global ResNet-18 embeddings and image-level scores; it does not localize defects into a pixel heatmap. The Team Approach page therefore shows frame scores and decisions, not heatmaps. Its exported clip contains sampled frames in order with original timestamps overlaid, not a frame-for-frame rendering of the entire source video. The two approaches remain separate and are not fused into one detector.
