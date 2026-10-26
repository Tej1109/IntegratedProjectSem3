from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torchvision import models, transforms

CATEGORIES = ["grid", "zipper", "hazelnut", "leather", "toothbrush"]
METHODS = {
    "kNN cosine": "knn_cosine",
    "Mahalanobis": "mahalanobis",
    "One-Class SVM": "ocsvm",
}
METHOD_LABELS = {v: k for k, v in METHODS.items()}
ACTIONS = ["skip", "prediction", "expert"]
RISK_LEVELS = ["low", "medium", "high"]
DEFAULT_THRESHOLD_PERCENTILE = 95
RISK_MARGIN = 0.10


@dataclass
class CategoryReference:
    category: str
    train_features: np.ndarray
    calibration_scores: dict[str, np.ndarray]
    fingerprint: str


@dataclass
class FittedDetector:
    method: str
    model: Any
    scaler: Any = None


def _device_name(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but PyTorch cannot see a CUDA device. Choose auto or cpu.")
    return device


def _get_encoder(device: str = "auto"):
    """Create the Step 1 ResNet-18 encoder, preferring the Colab-exported weights.

    The asset bundle includes the exact torchvision ImageNet ResNet-18 state dict
    (with the classification head removed), so local inference need not download
    weights at first launch. If that file is absent, torchvision's standard cached
    download path remains as a fallback.
    """
    device = _device_name(device)
    encoder = models.resnet18(weights=None)
    encoder.fc = nn.Identity()
    bundled_weights = Path(__file__).resolve().parent / "team_artifacts" / "vision" / "resnet18_imagenet1k_v1.pth"
    if bundled_weights.is_file():
        try:
            state = torch.load(bundled_weights, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(bundled_weights, map_location="cpu")
        encoder.load_state_dict(state, strict=True)
    else:
        # Backward-compatible fallback for users who have not generated the Colab bundle yet.
        pretrained = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        pretrained.fc = nn.Identity()
        encoder.load_state_dict(pretrained.state_dict(), strict=True)
        del pretrained
    encoder.eval().to(device)
    # Match Step 1: OpenCV INTER_AREA resize to 224x224 before ToTensor/normalization.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return encoder, transform, device


def _preprocess_rgb_array(rgb: np.ndarray, transform):
    """Apply the same resize/normalization sequence as the Step 1 vision notebook."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape HxWx3; got {rgb.shape}.")
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return transform(Image.fromarray(resized))


def _image_paths(folder: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()[:20]


def _embed_pil_images(images: list[Image.Image], device: str = "auto", batch_size: int = 32) -> np.ndarray:
    if not images:
        return np.empty((0, 512), dtype=np.float32)
    encoder, transform, device = _get_encoder(device)
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([
                _preprocess_rgb_array(np.asarray(img.convert("RGB")), transform)
                for img in images[start:start + batch_size]
            ]).to(device)
            result.append(encoder(batch).detach().float().cpu().numpy())
    return np.concatenate(result, axis=0).astype(np.float32)


def embed_rgb_frames(frames_rgb: list[np.ndarray], device: str = "auto", batch_size: int = 32) -> np.ndarray:
    if not frames_rgb:
        return np.empty((0, 512), dtype=np.float32)
    encoder, transform, device = _get_encoder(device)
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frames_rgb), batch_size):
            batch = torch.stack([
                _preprocess_rgb_array(frame, transform)
                for frame in frames_rgb[start:start + batch_size]
            ]).to(device)
            result.append(encoder(batch).detach().float().cpu().numpy())
    return np.concatenate(result, axis=0).astype(np.float32)


def embed_image_paths(paths: list[Path], device: str = "auto", batch_size: int = 32) -> np.ndarray:
    """Extract embeddings with the same OpenCV resize as the Step 1 notebook."""
    if not paths:
        return np.empty((0, 512), dtype=np.float32)
    encoder, transform, device = _get_encoder(device)
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as img:
                    batch_images.append(_preprocess_rgb_array(np.asarray(img.convert("RGB")), transform))
            batch = torch.stack(batch_images).to(device)
            result.append(encoder(batch).detach().float().cpu().numpy())
    return np.concatenate(result, axis=0).astype(np.float32)


def _fit_detector(method: str, train: np.ndarray) -> FittedDetector:
    if method == "knn_cosine":
        model = NearestNeighbors(n_neighbors=min(5, max(1, len(train))), metric="cosine")
        model.fit(train)
        return FittedDetector(method=method, model=model)
    if method == "mahalanobis":
        model = LedoitWolf().fit(train)
        return FittedDetector(method=method, model=model)
    if method == "ocsvm":
        scaler = StandardScaler().fit(train)
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(scaler.transform(train))
        return FittedDetector(method=method, model=model, scaler=scaler)
    raise ValueError(f"Unknown method: {method}. Choose one of {list(METHODS.values())}.")


def _raw_scores(fitted: FittedDetector, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if fitted.method == "knn_cosine":
        distances, _ = fitted.model.kneighbors(values)
        return distances.mean(axis=1).astype(np.float64)
    if fitted.method == "mahalanobis":
        squared = np.asarray(fitted.model.mahalanobis(values), dtype=np.float64)
        return np.sqrt(np.maximum(squared, 0.0))
    if fitted.method == "ocsvm":
        scaled = fitted.scaler.transform(values)
        return (-fitted.model.decision_function(scaled).reshape(-1)).astype(np.float64)
    raise ValueError(f"Unknown fitted method: {fitted.method}")


def _make_calibration_scores(train_features: np.ndarray) -> dict[str, np.ndarray]:
    if len(train_features) < 6:
        raise ValueError(
            f"Need at least 6 images in train/good for cross-validated calibration; found {len(train_features)}."
        )
    n_splits = min(5, len(train_features))
    out = {method: np.zeros(len(train_features), dtype=np.float64) for method in METHODS.values()}
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fit_idx, held_idx in splitter.split(train_features):
        fit_x, held_x = train_features[fit_idx], train_features[held_idx]
        for method in METHODS.values():
            detector = _fit_detector(method, fit_x)
            out[method][held_idx] = _raw_scores(detector, held_x)
    return out


def load_category_reference(
    data_root: str | Path,
    category: str,
    cache_root: str | Path = "team_artifacts/vision",
    device: str = "auto",
) -> CategoryReference:
    """Load cached ResNet features or build them from MVTec's category/train/good folder."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; choose one of {CATEGORIES}.")
    data_root = Path(data_root).expanduser().resolve()
    cache_dir = Path(cache_root).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Preferred route: reusable reference artifacts exported from the Colab builder.
    # This means the local application does not need the full MVTec dataset.
    exported_reference = cache_dir / f"reference_{category}.npz"
    if exported_reference.is_file():
        with np.load(exported_reference, allow_pickle=False) as saved:
            if "train_features" not in saved.files:
                raise ValueError(f"The exported reference {exported_reference} has no train_features array.")
            train_features = np.asarray(saved["train_features"], dtype=np.float32)
            missing_cal = [f"cal_{method}" for method in METHODS.values() if f"cal_{method}" not in saved.files]
            if missing_cal:
                raise ValueError(f"The exported reference {exported_reference} is missing calibration arrays: {missing_cal}")
            calibration = {method: np.asarray(saved[f"cal_{method}"], dtype=np.float64) for method in METHODS.values()}
            if "fingerprint" in saved.files:
                fingerprint_value = saved["fingerprint"]
                fingerprint = str(fingerprint_value.item() if np.asarray(fingerprint_value).ndim == 0 else fingerprint_value)
            else:
                fingerprint = "colab-export"
        if train_features.ndim != 2 or train_features.shape[1] != 512:
            raise ValueError(f"Invalid ResNet-18 reference shape in {exported_reference}: {train_features.shape}; expected (n, 512).")
        return CategoryReference(category, train_features, calibration, fingerprint)

    good_dir = data_root / category / "train" / "good"
    if not good_dir.is_dir():
        raise FileNotFoundError(
            f"No Colab-exported reference found at {exported_reference}, and the MVTec normal images were not found at {good_dir}.\n"
            "Recommended: run colab_asset_builder.ipynb in Google Colab and extract MVTec_Team_Assets.zip into your App3 project root. "
            "Alternatively, set the MVTec data root to the folder directly containing grid/, zipper/, hazelnut/, leather/, and toothbrush/."
        )
    paths = _image_paths(good_dir)
    if len(paths) < 6:
        raise ValueError(f"Only {len(paths)} images found in {good_dir}; at least 6 are needed for calibration.")
    fingerprint = _fingerprint(paths)
    cache_file = cache_dir / f"{category}_{fingerprint}.npz"
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as saved:
            features = saved["train_features"].astype(np.float32)
            calibration = {method: saved[f"cal_{method}"].astype(np.float64) for method in METHODS.values()}
        return CategoryReference(category, features, calibration, fingerprint)

    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    features = _embed_pil_images(images, device=device)
    calibration = _make_calibration_scores(features)
    payload = {"train_features": features}
    payload.update({f"cal_{method}": values for method, values in calibration.items()})
    np.savez_compressed(cache_file, **payload)
    return CategoryReference(category, features, calibration, fingerprint)


def score_embeddings(
    reference: CategoryReference,
    test_features: np.ndarray,
    method: str,
) -> list[dict[str, Any]]:
    if method not in METHODS.values():
        raise ValueError(f"Unknown method: {method}")
    fitted = _fit_detector(method, reference.train_features)
    raw_values = _raw_scores(fitted, test_features)
    ref_scores = np.sort(np.asarray(reference.calibration_scores[method], dtype=np.float64))
    if not len(ref_scores):
        raise ValueError(f"No calibration scores available for method {method}.")
    raw_threshold = float(np.percentile(ref_scores, DEFAULT_THRESHOLD_PERCENTILE))
    norm_scores = np.searchsorted(ref_scores, raw_values, side="left") / len(ref_scores)
    norm_threshold = float(np.searchsorted(ref_scores, raw_threshold, side="left") / len(ref_scores))
    return [
        {
            # Keep the Step 1 CSV semantics: anomaly_score is raw; normalized_score is 0..1.
            "raw_score": float(raw),
            "anomaly_score": float(raw),
            "normalized_score": float(np.clip(norm, 0.0, 1.0)),
            "threshold": raw_threshold,
            "normalized_threshold": norm_threshold,
            "prediction": int(raw >= raw_threshold),
            "prediction_label": "ANOMALY" if raw >= raw_threshold else "NORMAL",
            "risk_level": assign_risk_level(float(norm), norm_threshold),
        }
        for raw, norm in zip(raw_values, norm_scores)
    ]


def assign_risk_level(normalized_score: float, normalized_threshold: float = 0.95) -> str:
    margin = float(normalized_score) - float(normalized_threshold)
    if margin < -RISK_MARGIN:
        return "low"
    if margin < 0:
        return "medium"
    return "high"


def load_q_policy(
    policy_path: str | Path,
    expected_method: str = "mahalanobis",
) -> dict[tuple[str, str], str]:
    """Load the shared Step 2 Q-table only with the vision method it was trained for.

    The supplied RL notebook filters Step 1 outputs to Mahalanobis records. The
    other two Step 1 detectors remain available in the UI, but are vision-only.
    """
    path = Path(policy_path).expanduser()
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    trained_method = metadata.get("vision_method") if isinstance(metadata, dict) else None
    if trained_method != expected_method:
        shown_method = trained_method or "unknown / missing metadata"
        raise ValueError(
            f"This Q-learning policy was trained for {shown_method!r}, but the selected "
            f"Step 2-compatible scheduler requires {expected_method!r}. "
            "Run rebuild_mahalanobis_policy.py from the project root if "
            "team_artifacts/outputs/vision/current_anomaly_score.csv exists, or refresh "
            "the asset bundle using the updated Colab notebook."
        )
    policy: dict[tuple[str, str], str] = {}
    for row in payload.get("states", []):
        category = str(row.get("category", ""))
        risk = str(row.get("risk_level", ""))
        values = row.get("q_values", [])
        if len(values) != len(ACTIONS):
            continue
        numeric = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(numeric)):
            continue
        best_idx = int(np.argmax(numeric))
        if 0 <= best_idx < len(ACTIONS):
            policy[(category, risk)] = ACTIONS[best_idx]
    return policy


def choose_q_action(policy: dict[tuple[str, str], str], category: str, risk_level: str) -> str | None:
    """Return the greedy action. None means this category/risk state was not trained."""
    return policy.get((category, risk_level))


def describe_action(action: str | None) -> str:
    return {
        "skip": "Skip further inspection",
        "prediction": "Use the model prediction",
        "expert": "Request expert review",
        None: "No trained action for this state",
    }.get(action, "No trained action for this state")


def expected_policy_path(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "team_artifacts" / "rl" / "shared_q_policy.json"


def expected_scores_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    candidates = [
        root / "outputs" / "vision" / "current_anomaly_score.csv",
        root / "team_artifacts" / "outputs" / "vision" / "current_anomaly_score.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
