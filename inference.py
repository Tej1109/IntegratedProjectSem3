from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision import models, transforms
from ultralytics import YOLO

CATEGORIES = ["grid", "zipper", "hazelnut", "leather", "toothbrush"]
CATEGORY_DETECTORS = {
    "grid": "STFPM · ResNet-18",
    "zipper": "YOLO26 + PatchCore-style local k-NN",
    "hazelnut": "YOLO26 + PatchCore-style local k-NN",
    "leather": "YOLO26 + PatchCore-style local k-NN",
    "toothbrush": "YOLO26 + PatchCore-style local k-NN",
}
ACTION_NAMES = {0: "DO_NOTHING", 1: "FLAG_ANOMALY", 2: "HUMAN_REVIEW"}


class WideResNetPatchExtractor(nn.Module):
    """Same Wide-ResNet50-2 layer2/layer3 patch extractor as the training notebook."""
    def __init__(self, pretrained: bool = False):
        super().__init__()
        base = models.wide_resnet50_2(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.output_dim = 1536
        del base

    def forward(self, x):
        x = self.stem(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f2 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        f = F.adaptive_avg_pool2d(f, (7, 7))
        return F.normalize(f, p=2, dim=1)


class STFPMResNet18(nn.Module):
    """Exact student/teacher network structure from the notebook."""
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet18(weights=None)

    def forward(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        f1 = b.layer1(x)
        f2 = b.layer2(f1)
        f3 = b.layer3(f2)
        return (f1, f2, f3)


class CausalTemporalTransformer(nn.Module):
    """Architecture definition copied from MVTEC_FINAL.ipynb."""
    def __init__(self, input_dim=6, d_model=32, nhead=4, num_layers=2, max_len=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x):
        t = x.shape[1]
        z = self.input_proj(x) + self.pos_embed[:, :t, :]
        causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        z = self.encoder(z, mask=causal_mask)
        return self.head(self.norm(z))


def ecdf_percentile(reference, value) -> float:
    reference = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    value = float(np.asarray(value).reshape(-1)[0])
    return float(np.searchsorted(reference, value, side="right") / max(len(reference), 1))


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class HybridInspectionEngine:
    """Loads the user's saved hybrid detector, temporal transformer and PPO policy."""
    def __init__(self, project_dir: str | Path, device: str = "auto", enable_subtypes: bool = False):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.models_dir = self.project_dir / "models"
        self.assets_dir = self.project_dir / "assets"
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is False. Choose auto or cpu.")
            self.device = torch.device(device)
        self.enable_subtypes = enable_subtypes
        self._subtype_ready = False

        required = [
            self.models_dir / "yolo26n.pt",
            self.models_dir / "wide_resnet50_2_patch_extractor.pth",
            self.models_dir / "stfpm_teacher_resnet18_grid.pth",
            self.models_dir / "stfpm_student_resnet18_grid.pth",
            self.models_dir / "causal_temporal_transformer.pt",
            self.models_dir / "temporal_transformer_input_scaler.joblib",
            self.assets_dir / "stfpm_calibration_grid.npz",
        ]
        self.v4_policy_path = self.models_dir / "ppo_monitor_flag_escalate_recall_v4toothbrush.zip"
        self.legacy_policy_path = self.models_dir / "ppo_inspection_policy.zip"
        if not self.v4_policy_path.is_file() and not self.legacy_policy_path.is_file():
            # Prefer V4; permit the legacy PPO only for backwards compatibility.
            required.append(self.v4_policy_path)
        for cat in CATEGORIES:
            required.extend([
                self.models_dir / f"yolo26_embedding_knn_{cat}.joblib",
                self.models_dir / f"patchcore_patch_bank_{cat}.pt",
                self.assets_dir / f"calibration_{cat}.npz",
            ])
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            preview = "\n".join(f"• {p}" for p in missing[:12])
            suffix = f"\n…plus {len(missing)-12} more" if len(missing) > 12 else ""
            raise FileNotFoundError(
                "The exported MVTec project directory is incomplete. Expected folders named "
                "'models/' and 'assets/' under the selected project root. Missing files:\n"
                + preview + suffix
            )

        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.stfpm_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.subtype_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # YOLO26 embedding backbone
        self.yolo = YOLO(str(self.models_dir / "yolo26n.pt"))
        self.yolo.to(str(self.device))

        # Patch feature extractor
        self.patch_extractor = WideResNetPatchExtractor(pretrained=False).to(self.device).eval()
        patch_state = safe_torch_load(self.models_dir / "wide_resnet50_2_patch_extractor.pth", "cpu")
        self.patch_extractor.load_state_dict(patch_state, strict=True)
        for parameter in self.patch_extractor.parameters():
            parameter.requires_grad_(False)

        self.knn_models = {}
        self.patch_banks = {}
        self.legacy_calibrations = {}
        for cat in CATEGORIES:
            self.knn_models[cat] = joblib.load(self.models_dir / f"yolo26_embedding_knn_{cat}.joblib")
            bank = safe_torch_load(self.models_dir / f"patchcore_patch_bank_{cat}.pt", self.device)
            if isinstance(bank, dict) and "bank" in bank:
                bank = bank["bank"]
            bank = torch.as_tensor(bank, dtype=torch.float32, device=self.device)
            self.patch_banks[cat] = F.normalize(bank, p=2, dim=1)
            with np.load(self.assets_dir / f"calibration_{cat}.npz", allow_pickle=False) as cal:
                self.legacy_calibrations[cat] = {
                    "yolo_cal_raw": np.asarray(cal["yolo_cal_raw"], dtype=np.float32),
                    "patch_cal_raw": np.asarray(cal["patch_cal_raw"], dtype=np.float32),
                    "ensemble_cal_raw": np.asarray(cal["ensemble_cal_raw"], dtype=np.float32),
                }

        # STFPM grid models and normal-only calibration references
        self.stfpm_teacher = STFPMResNet18(pretrained=False).to(self.device).eval()
        self.stfpm_student = STFPMResNet18(pretrained=False).to(self.device).eval()
        self.stfpm_teacher.load_state_dict(safe_torch_load(self.models_dir / "stfpm_teacher_resnet18_grid.pth", "cpu"), strict=True)
        self.stfpm_student.load_state_dict(safe_torch_load(self.models_dir / "stfpm_student_resnet18_grid.pth", "cpu"), strict=True)
        for p in self.stfpm_teacher.parameters():
            p.requires_grad_(False)
        for p in self.stfpm_student.parameters():
            p.requires_grad_(False)
        with np.load(self.assets_dir / "stfpm_calibration_grid.npz", allow_pickle=False) as cal:
            self.stfpm_cal = {k: np.asarray(cal[k], dtype=np.float32) for k in cal.files}

        # Temporal Transformer + fitted input scaler
        self.temporal_scaler = joblib.load(self.models_dir / "temporal_transformer_input_scaler.joblib")
        transformer_ckpt = safe_torch_load(self.models_dir / "causal_temporal_transformer.pt", "cpu")
        self.temporal = CausalTemporalTransformer(
            input_dim=int(transformer_ckpt.get("input_dim", 6)),
            d_model=int(transformer_ckpt.get("d_model", 32)),
            nhead=int(transformer_ckpt.get("nhead", 4)),
            num_layers=int(transformer_ckpt.get("num_layers", 2)),
            max_len=int(transformer_ckpt.get("max_len", 128)),
            dropout=0.1,
        ).to(self.device)
        self.temporal.load_state_dict(transformer_ckpt["state_dict"], strict=True)
        self.temporal.eval()
        self.temporal_max_len = int(transformer_ckpt.get("max_len", 128))
        self.temporal_trained_seq_len = int(transformer_ckpt.get("seq_len", 24))

        # Prefer the standalone V4 recall-balanced policy. It has a distinct 20-feature
        # observation contract and must not be fed the legacy Transformer's 10-feature vector.
        # Fall back only when V4 is absent, and expose the selected policy in the UI/report.
        from stable_baselines3 import PPO
        if self.v4_policy_path.is_file():
            self.ppo = PPO.load(str(self.v4_policy_path), device="cpu")
            self.ppo_mode = "v4_recall_balanced"
            self.ppo_model_filename = self.v4_policy_path.name
            candidate_configs = [
                self.project_dir / "results" / "ppo_monitor_flag_escalate_recall_v4toothbrush_config.json",
                self.models_dir / "ppo_monitor_flag_escalate_recall_v4toothbrush_config.json",
                self.project_dir / "ppo_monitor_flag_escalate_recall_v4toothbrush_config.json",
            ]
            self.ppo_v4_config = {}
            for config_path in candidate_configs:
                if config_path.is_file():
                    try:
                        self.ppo_v4_config = json.loads(config_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        raise RuntimeError(f"Could not read V4 PPO config at {config_path}: {exc}") from exc
                    break
            obs_shape = tuple(getattr(self.ppo.observation_space, "shape", ()) or ())
            if obs_shape != (20,):
                raise RuntimeError(
                    f"V4 PPO checkpoint has observation shape {obs_shape}, expected (20,). "
                    "Check that the file in models/ is ppo_monitor_flag_escalate_recall_v4toothbrush.zip."
                )
        else:
            self.ppo = PPO.load(str(self.legacy_policy_path), device="cpu")
            self.ppo_mode = "legacy_transformer"
            self.ppo_model_filename = self.legacy_policy_path.name
            self.ppo_v4_config = {}

        manifest = self.project_dir / "model_manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.is_file() else {}

    def _as_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    @torch.inference_mode()
    def _stfpm_score(self, image: Image.Image, original_shape: tuple[int, int]):
        x = self.stfpm_transform(image).unsqueeze(0).to(self.device)
        teacher_features = self.stfpm_teacher(x)
        student_features = self.stfpm_student(x)
        residual_maps = []
        for tf, sf in zip(teacher_features, student_features):
            tu = F.normalize(tf, p=2, dim=1)
            su = F.normalize(sf, p=2, dim=1)
            residual = (tu - su).pow(2).mean(dim=1, keepdim=True)
            residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
            residual_maps.append(residual)
        amap = torch.stack(residual_maps, dim=0).mean(dim=0)[0, 0]
        flat = amap.flatten()
        n_pixels = flat.numel()
        k1 = max(1, int(np.ceil(0.01 * n_pixels)))
        k5 = max(k1, int(np.ceil(0.05 * n_pixels)))
        top = torch.topk(flat, k=k5).values
        top1_mean = float(top[:k1].mean().cpu())
        top5_mean = float(top.mean().cpu())
        raw = 0.65 * top1_mean + 0.35 * top5_mean
        map_np = amap.detach().float().cpu().numpy()
        stats = {
            "map_mean": float(map_np.mean()),
            "map_std": float(map_np.std()),
            "map_max": float(map_np.max()),
            "map_top1_mean": top1_mean,
            "map_top5_mean": top5_mean,
        }
        score = ecdf_percentile(self.stfpm_cal["normal_score_raw"], raw)
        stat_pct = {
            key: ecdf_percentile(self.stfpm_cal[f"normal_{key}"], val)
            for key, val in stats.items()
        }
        global_e = stat_pct["map_mean"]
        local_e = stat_pct["map_top5_mean"]
        dispersion = stat_pct["map_std"]
        max_e = stat_pct["map_max"]
        h, w = original_shape
        heat = cv2.resize(map_np, (w, h), interpolation=cv2.INTER_CUBIC)
        heat = np.maximum(heat, 0)
        return {
            "detector": "STFPM-ResNet18",
            "raw_score": float(raw),
            "anomaly_score": float(score),
            "global_evidence_pct": float(global_e),
            "local_evidence_pct": float(local_e),
            "evidence_dispersion_pct": float(dispersion),
            "evidence_max_pct": float(max_e),
            "heatmap": heat,
            "details": {f"stfpm_{k}": float(v) for k, v in stats.items()},
        }

    def _yolo_embedding(self, image: Image.Image) -> np.ndarray:
        # Windows-safe temporary image handling: close the file before PIL/YOLO reopen it.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp_path = temp.name
        try:
            image.save(temp_path, format="JPEG", quality=95)
            out = self.yolo.embed(
                source=[temp_path], imgsz=224, batch=1,
                device=str(self.device), verbose=False
            )
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
        if torch.is_tensor(out):
            arr = out.detach().float().cpu().numpy()
            if arr.ndim == 1:
                arr = arr[None, :]
            arr = arr.reshape(arr.shape[0], -1)
        elif isinstance(out, (list, tuple)):
            if len(out) == 1:
                value = out[0]
                if torch.is_tensor(value):
                    arr = value.detach().float().cpu().numpy().reshape(1, -1)
                else:
                    arr = np.asarray(value, dtype=np.float32).reshape(1, -1)
            else:
                arr = np.stack([
                    x.detach().float().cpu().numpy().reshape(-1) if torch.is_tensor(x)
                    else np.asarray(x, dtype=np.float32).reshape(-1)
                    for x in out
                ])
        else:
            arr = np.asarray(out, dtype=np.float32).reshape(1, -1)
        if arr.shape[0] != 1:
            raise RuntimeError(f"Unexpected YOLO embed output shape {arr.shape}")
        return arr[0].astype(np.float32)

    @torch.inference_mode()
    def _legacy_score(self, image: Image.Image, category: str, original_shape: tuple[int, int]):
        embedding = self._yolo_embedding(image)
        yolo_raw = float(self.knn_models[category].kneighbors(normalize(embedding[None, :]), return_distance=True)[0][0, 0])
        x = self.image_transform(image).unsqueeze(0).to(self.device)
        grid = self.patch_extractor(x).permute(0, 2, 3, 1).reshape(1, 49, -1)
        grid = F.normalize(grid, p=2, dim=-1)
        bank = self.patch_banks[category]
        sim = torch.matmul(grid, bank.T)
        distances = (1.0 - sim.max(dim=-1).values).clamp(min=0.0)
        top = torch.topk(distances, k=min(5, distances.shape[1]), dim=1).values
        patch_raw = float(top.mean().cpu())
        cal = self.legacy_calibrations[category]
        yolo_pct = ecdf_percentile(cal["yolo_cal_raw"], yolo_raw)
        patch_pct = ecdf_percentile(cal["patch_cal_raw"], patch_raw)
        ensemble_raw = 0.30 * yolo_pct + 0.70 * patch_pct
        score = ecdf_percentile(cal["ensemble_cal_raw"], ensemble_raw)
        dmap = distances[0].detach().cpu().numpy().reshape(7, 7)
        h, w = original_shape
        heat = cv2.resize(dmap, (w, h), interpolation=cv2.INTER_CUBIC)
        return {
            "detector": "YOLO26 + PatchCore-style k-NN",
            "raw_score": float(ensemble_raw),
            "anomaly_score": float(score),
            "global_evidence_pct": float(yolo_pct),
            "local_evidence_pct": float(patch_pct),
            "evidence_dispersion_pct": float(abs(yolo_pct - patch_pct)),
            "evidence_max_pct": float(max(yolo_pct, patch_pct)),
            "heatmap": heat,
            "details": {"yolo_raw_score": yolo_raw, "patchcore_raw_score": patch_raw,
                        "yolo_percentile": yolo_pct, "patchcore_percentile": patch_pct},
        }

    def score_frame(self, frame_bgr: np.ndarray, category: str):
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category {category}. Choose one of {CATEGORIES}.")
        image = self._as_pil(frame_bgr)
        height, width = frame_bgr.shape[:2]
        if category == "grid":
            return self._stfpm_score(image, (height, width))
        return self._legacy_score(image, category, (height, width))

    def _apply_v4_policy(self, rows, scores, score_delta, temporal_probs):
        """Replay PPO V4's exact 20-feature state representation, without guardrail overrides."""
        cfg = self.ppo_v4_config or {}
        fps = max(1, int(cfg.get("simulated_fps", 4)))
        lookback_frames = max(1, int(round(float(cfg.get("lookback_seconds", 3)) * fps)))
        persistence_frames = max(1, int(round(float(cfg.get("persistence_seconds", 5)) * fps)))
        cooldown_frames = max(1, int(round(float(cfg.get("review_cooldown_seconds", 10)) * fps)))
        threshold = float(cfg.get("score_threshold_context", 0.95))
        # These constants match the v3/v4 training cell's derive_temporal_signals().
        surge_min_delta = float(cfg.get("surge_min_delta", 0.20))
        surge_std_multiplier = float(cfg.get("surge_std_multiplier", 2.0))
        std_floor = float(cfg.get("std_floor", 0.03))
        action_names = {0: "MONITOR", 1: "FLAG_ANOMALY", 2: "HUMAN_REVIEW"}

        score_history: list[float] = []
        action_history: list[int] = []
        cooldown_remaining = 0
        out = []
        for i, row in enumerate(rows):
            score = float(np.clip(scores[i], 0.0, 1.0))
            hist_window = score_history[-lookback_frames:]
            if hist_window:
                previous_mean = float(np.mean(hist_window))
                previous_std = float(np.std(hist_window, ddof=0))
                delta = float(score - score_history[-1])
                jump_z = float(np.clip((score - previous_mean) / max(previous_std, std_floor), -8, 8))
                delta_3s = float(score - (score_history[-lookback_frames] if len(score_history) >= lookback_frames else score_history[0]))
            else:
                previous_mean, previous_std, delta, jump_z, delta_3s = score, 0.0, 0.0, 0.0, 0.0

            current_window = (score_history + [score])[-lookback_frames:]
            high_fraction = float(np.mean(np.asarray(current_window, dtype=np.float32) >= threshold)) if current_window else 0.0
            run = 1 if score >= threshold else 0
            if run:
                for old in reversed(score_history):
                    if old >= threshold:
                        run += 1
                    else:
                        break
            persistent = bool(score >= threshold and run >= persistence_frames)
            sharp_rise = bool(score >= threshold and delta >= max(surge_min_delta, surge_std_multiplier * previous_std))
            escalation_candidate = bool(sharp_rise or persistent)

            # The app's score_frame uses these names; training used the aliases below.
            ev_local = row.get("evidence_local_pct", row.get("local_evidence_pct", score))
            ev_global = row.get("evidence_global_pct", row.get("global_evidence_pct", score))
            ev_disp = row.get("evidence_dispersion_pct", abs(float(ev_local) - float(ev_global)))
            ev_max = row.get("evidence_max_pct", max(float(ev_local), float(ev_global)))
            def clamp01(value, fallback):
                try:
                    value = float(value)
                    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else float(fallback)
                except Exception:
                    return float(fallback)
            ev_local = clamp01(ev_local, score)
            ev_global = clamp01(ev_global, score)
            ev_disp = clamp01(ev_disp, abs(ev_local - ev_global))
            ev_max = clamp01(ev_max, max(ev_local, ev_global))

            recent_actions = action_history[-lookback_frames:]
            prev_action_oh = [0.0, 0.0, 0.0]
            if recent_actions:
                prev_action_oh[int(recent_actions[-1])] = 1.0
            flag_fraction = float(sum(int(a) == 1 for a in recent_actions) / lookback_frames)
            cooldown_fraction = float(min(max(cooldown_remaining, 0) / cooldown_frames, 1.0))
            cooldown_active = cooldown_remaining > 0

            obs = np.asarray([
                score,
                float(np.clip(delta, -1, 1)),
                jump_z,
                float(np.clip(previous_mean, 0, 1)),
                float(np.clip(previous_std, 0, 1)),
                high_fraction,
                float(min(run / persistence_frames, 1.0)),
                float(np.clip(delta_3s, -1, 1)),
                float(sharp_rise),
                float(persistent),
                float(escalation_candidate),
                ev_local,
                ev_global,
                ev_disp,
                ev_max,
                prev_action_oh[0],
                prev_action_oh[1],
                prev_action_oh[2],
                flag_fraction,
                cooldown_fraction,
            ], dtype=np.float32)
            if obs.shape != (20,):
                raise RuntimeError(f"Internal V4 observation shape is {obs.shape}, expected (20,).")
            action_arr, _ = self.ppo.predict(obs, deterministic=True)
            action = int(np.asarray(action_arr).item())
            if action not in action_names:
                raise RuntimeError(f"V4 PPO returned unsupported action id {action}.")
            review_event = bool(action == 2 and not cooldown_active)

            item = dict(row)
            item.update({
                "score_delta": float(np.clip(delta, -1, 1)),
                "temporal_anomaly_probability": float(temporal_probs[i]),
                "ppo_raw_action": action_names[action],
                "ppo_final_action": action_names[action],
                "ppo_policy": "PPO V4 recall-balanced",
                "ppo_checkpoint": self.ppo_model_filename,
                "ppo_decision_source": "learned_v4_policy_no_inference_override",
                "human_review_guardrail": False,
                "human_review_surge_guardrail": False,
                "human_review_persistence_guardrail": False,
                "human_review_guardrail_reason": "",
                "human_review_event": review_event,
                "review_cooldown_active_before_action": bool(cooldown_active),
                "previous_3s_mean_score": float(previous_mean),
                "previous_3s_std_score": float(previous_std),
                "jump_z_score": float(jump_z),
                "delta_vs_3s_ago": float(np.clip(delta_3s, -1, 1)),
                "high_score_run_frames": int(run),
                "sharp_rise_signal": bool(sharp_rise),
                "persistent_high_5s_signal": bool(persistent),
                "escalation_candidate_signal": bool(escalation_candidate),
                "high_score_count_last_5": int(np.sum(np.asarray(scores[max(0, i-4):i+1]) >= 0.97)),
                "rl_observation": obs,
            })
            out.append(item)

            # Match training rollout semantics: review is a non-terminal event with cooldown.
            if review_event:
                cooldown_remaining = cooldown_frames
            elif cooldown_remaining > 0:
                cooldown_remaining -= 1
            action_history.append(action)
            score_history.append(score)
            keep = max(lookback_frames, persistence_frames, cooldown_frames) + 1
            if len(score_history) > keep:
                score_history = score_history[-keep:]
            if len(action_history) > keep:
                action_history = action_history[-keep:]
        return out

    def temporal_and_rl(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run the trained temporal Transformer and PPO with the notebook's 6/10-feature order."""
        if not rows:
            return []
        s = np.asarray([r["anomaly_score"] for r in rows], dtype=np.float32)
        local = np.asarray([r["local_evidence_pct"] for r in rows], dtype=np.float32)
        global_e = np.asarray([r["global_evidence_pct"] for r in rows], dtype=np.float32)
        dispersion = np.asarray([r["evidence_dispersion_pct"] for r in rows], dtype=np.float32)
        max_e = np.asarray([r["evidence_max_pct"] for r in rows], dtype=np.float32)
        delta = np.diff(s, prepend=s[0]).astype(np.float32)
        mean3 = np.asarray([s[max(0, i-2):i+1].mean() for i in range(len(s))], dtype=np.float32)
        std5 = np.asarray([s[max(0, i-4):i+1].std() for i in range(len(s))], dtype=np.float32)
        delta3 = np.asarray([s[i] - s[max(0, i-3)] for i in range(len(s))], dtype=np.float32)
        max5 = np.asarray([s[max(0, i-4):i+1].max() for i in range(len(s))], dtype=np.float32)
        frac_hi = np.asarray([(s[max(0, i-4):i+1] >= 0.95).mean() for i in range(len(s))], dtype=np.float32)

        # Same six Transformer features used when training the project.
        tx_raw = np.column_stack([s, local, global_e, delta, mean3, std5]).astype(np.float32)
        tx_scaled = self.temporal_scaler.transform(tx_raw).astype(np.float32)
        window_len = min(24, self.temporal_max_len)
        temporal_probs = []
        # Use a causal rolling context. The original Transformer was trained with 24-step sequences.
        with torch.inference_mode():
            for t in range(len(rows)):
                start = max(0, t - window_len + 1)
                x = torch.from_numpy(tx_scaled[start:t+1][None, :, :]).to(self.device)
                logits = self.temporal(x)
                prob = torch.softmax(logits, dim=-1)[0, -1, 1].item()
                temporal_probs.append(float(prob))

        if self.ppo_mode == "v4_recall_balanced":
            return self._apply_v4_policy(rows, s, delta, temporal_probs)

        out = []
        for i, row in enumerate(rows):
            obs = np.asarray([
                s[i], delta[i], mean3[i], std5[i], temporal_probs[i],
                local[i], global_e[i], delta3[i], max5[i], frac_hi[i],
            ], dtype=np.float32)
            raw_action, _ = self.ppo.predict(obs, deterministic=True)
            raw_action = int(np.asarray(raw_action).item())

            # Guardrail A: an abrupt rise into a very high anomaly-score range.
            surge_guardrail = bool(s[i] >= 0.97 and delta[i] >= 0.20)

            # Guardrail B: persistent high anomaly scores. A review is requested
            # when the current score is >= 0.97 and at least 3 of the latest
            # 5 sampled frames (including this frame) are also >= 0.97.
            w_start = max(0, i - 4)
            high_score_count = int(np.sum(s[w_start:i + 1] >= 0.97))
            persistence_guardrail = bool(s[i] >= 0.97 and high_score_count >= 3)

            guardrail = bool(surge_guardrail or persistence_guardrail)
            final_action = 2 if guardrail else raw_action
            if surge_guardrail and persistence_guardrail:
                guardrail_reason = "SCORE_SURGE_AND_PERSISTENT_HIGH_SCORE"
            elif surge_guardrail:
                guardrail_reason = "SCORE_SURGE"
            elif persistence_guardrail:
                guardrail_reason = "PERSISTENT_HIGH_SCORE"
            else:
                guardrail_reason = ""

            item = dict(row)
            item.update({
                "score_delta": float(delta[i]),
                "temporal_anomaly_probability": float(temporal_probs[i]),
                "ppo_raw_action": ACTION_NAMES[raw_action],
                "ppo_final_action": ACTION_NAMES[final_action],
                "human_review_guardrail": guardrail,
                "human_review_surge_guardrail": surge_guardrail,
                "human_review_persistence_guardrail": persistence_guardrail,
                "human_review_guardrail_reason": guardrail_reason,
                "high_score_count_last_5": high_score_count,
                "rl_observation": obs,
            })
            out.append(item)
        return out

    def predict_subtype(self, frame_bgr: np.ndarray, category: str) -> dict[str, Any] | None:
        """Optional DINOv2 subtype classifier. Lazy-loads DINOv2 only on demand."""
        clf_path = self.models_dir / f"defect_subtype_{category}_dinov2_vits14_linearsvm.joblib"
        backbone_path = self.models_dir / "dinov2_vits14_subtype_backbone.pth"
        if not clf_path.is_file() or not backbone_path.is_file():
            return None
        if not self._subtype_ready:
            try:
                # Requires network access only if torch.hub has not cached DINOv2 source.
                self.subtype_backbone = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", pretrained=False, trust_repo=True
                ).to(self.device).eval()
                state = safe_torch_load(backbone_path, "cpu")
                self.subtype_backbone.load_state_dict(state, strict=True)
                self._subtype_ready = True
            except Exception as exc:
                raise RuntimeError(
                    "DINOv2 optional subtype model could not be initialized. Check internet access "
                    "for first-time torch.hub setup and verify dinov2_vits14_subtype_backbone.pth. "
                    f"Original error: {exc}"
                ) from exc
        clf = joblib.load(clf_path)
        image = self._as_pil(frame_bgr)
        x = self.subtype_transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            d = self.subtype_backbone.forward_features(x)
            cls = F.normalize(d["x_norm_clstoken"], p=2, dim=1)
            patch_mean = F.normalize(d["x_norm_patchtokens"].mean(dim=1), p=2, dim=1)
            feature = torch.cat([cls, patch_mean], dim=1).cpu().numpy()
        pred = str(clf.predict(feature)[0])
        confidence = None
        if hasattr(clf, "decision_function"):
            try:
                dec = np.asarray(clf.decision_function(feature)).reshape(-1)
                confidence = float(np.max(dec)) if dec.size else None
            except Exception:
                pass
        return {"predicted_subtype": pred, "decision_margin": confidence}


def blend_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Blend a continuous anomaly map over the original BGR image."""
    h, w = frame_bgr.shape[:2]
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    low, high = float(np.nanmin(hm)), float(np.nanmax(hm))
    norm = (hm - low) / (high - low + 1e-8)
    heat_u8 = np.clip(norm * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, colored, alpha, 0)
