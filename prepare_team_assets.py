from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from team_inference import (
    CATEGORIES, METHODS as VISION_METHODS, embed_image_paths,
    load_category_reference, score_embeddings,
)

ACTIONS = ["skip", "prediction", "expert"]
RISK_LEVELS = ["low", "medium", "high"]
ACTION_COSTS = {"skip": 0, "prediction": 1, "expert": 5}
MISSED_DEFECT_PENALTY = 10
FALSE_REJECTION_PENALTY = 3
ALPHA = 0.1
GAMMA = 0.9  # independent records are terminal, so no future reward is added
EPSILON = 0.2
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05
EPISODES = 500
SEED = 42
RISK_MARGIN = 0.10


def _binary(value) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"anomaly", "defective", "defect", "true", "yes"}:
            return 1
        if value in {"normal", "good", "false", "no"}:
            return 0
    return int(float(value))


def assign_risk(row) -> str:
    margin = float(row["normalized_score"]) - float(row["normalized_threshold"])
    if margin < -RISK_MARGIN:
        return "low"
    if margin < 0:
        return "medium"
    return "high"


def reward_for(row, action: str) -> float:
    label = int(row["label"])
    pred = int(row["prediction"])
    reward = -ACTION_COSTS[action]
    if action == "skip":
        missed = label == 1
        false_rejection = False
    elif action == "prediction":
        missed = label == 1 and pred == 0
        false_rejection = label == 0 and pred == 1
    elif action == "expert":
        # Mirrors the notebook's explicitly simulated perfect expert assumption.
        missed = False
        false_rejection = False
    else:
        raise ValueError(action)
    if missed:
        reward -= MISSED_DEFECT_PENALTY
    if false_rejection:
        reward -= FALSE_REJECTION_PENALTY
    return reward


def _image_paths(folder: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def build_score_csv_from_mvtec(mvtec_root: Path, output_csv: Path, device: str) -> Path:
    """Recreate Step 1's score export from MVTec folders, without executing the notebook."""
    mvtec_root = mvtec_root.expanduser().resolve()
    if not mvtec_root.is_dir():
        raise FileNotFoundError(
            f"MVTec data root not found: {mvtec_root}. It must directly contain {', '.join(CATEGORIES)}."
        )
    cache_root = Path("team_artifacts/vision").resolve()
    rows: list[dict] = []
    total_test_images = 0
    for category in CATEGORIES:
        test_root = mvtec_root / category / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(f"Expected MVTec test split at: {test_root}")
        paths = _image_paths(test_root)
        if not paths:
            raise ValueError(f"No test images found under {test_root}")
        total_test_images += len(paths)
        print(f"[{category}] loading normal reference and embedding {len(paths)} test images…")
        reference = load_category_reference(mvtec_root, category, cache_root=cache_root, device=device)
        features = embed_image_paths(paths, device=device)
        relative_paths = [p.relative_to(mvtec_root).as_posix() for p in paths]
        labels = [int(p.parent.name.lower() != "good") for p in paths]
        defect_types = [p.parent.name for p in paths]
        for method in VISION_METHODS.values():
            scored = score_embeddings(reference, features, method)
            for rel_path, label, defect_type, item in zip(relative_paths, labels, defect_types, scored):
                rows.append({
                    "category": category,
                    "image_path": rel_path,
                    "true_defect_type": defect_type,
                    "label": label,
                    "method": method,
                    "anomaly_score": float(item["raw_score"]),
                    "threshold": float(item["threshold"]),
                    "prediction": int(item["prediction"]),
                    # The optional notebook defect-type classifier is intentionally not trained here.
                    "predicted_defect_type": "not_run_by_app3_asset_builder",
                    "normalized_score": float(item["normalized_score"]),
                    "normalized_threshold": float(item["normalized_threshold"]),
                })
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    score_df = pd.DataFrame(rows)
    score_df.to_csv(output_csv, index=False)
    print(f"Saved {len(score_df):,} score rows from {total_test_images:,} test images across {len(CATEGORIES)} categories and {len(VISION_METHODS)} methods to {output_csv}")
    return output_csv


def load_policy_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find Step 1 output: {csv_path}\n"
            "Run the Step 1 vision notebook first, or pass --scores-csv with its current_anomaly_score.csv file."
        )
    df = pd.read_csv(csv_path)
    required = {"category", "image_path", "label", "method", "prediction", "normalized_score", "normalized_threshold"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"The CSV is missing columns required by the notebook's RL scheduler: {missing}")
    df = df[df["method"].astype(str).str.lower().eq("mahalanobis")].copy()
    if df.empty:
        raise ValueError("No rows with method='mahalanobis' were found in the Step 1 score CSV.")
    df["label"] = df["label"].map(_binary).astype(int)
    df["prediction"] = df["prediction"].map(_binary).astype(int)
    df["category"] = df["category"].astype(str).str.lower()
    df = df[df["category"].isin(CATEGORIES)].copy()
    df["risk_level"] = df.apply(assign_risk, axis=1)
    df["state"] = list(zip(df["category"], df["risk_level"]))
    if df["image_path"].isna().any():
        raise ValueError("The Step 1 output contains missing image_path values.")
    return df.reset_index(drop=True)


def train_q_table(df: pd.DataFrame) -> tuple[list[tuple[str, str]], np.ndarray, int, int]:
    stratify_key = df["category"].astype(str) + "_" + df["label"].astype(str)
    try:
        train_df, heldout_df = train_test_split(
            df, test_size=0.20, random_state=SEED, stratify=stratify_key
        )
    except ValueError:
        # Small or incomplete datasets may not support all stratum counts.
        train_df, heldout_df = train_test_split(df, test_size=0.20, random_state=SEED)

    # State order mirrors the notebook's states created before its 80/20 split.
    states = sorted(set(df["state"].tolist()))
    state_to_index = {state: i for i, state in enumerate(states)}
    action_to_index = {action: i for i, action in enumerate(ACTIONS)}
    q_table = np.zeros((len(states), len(ACTIONS)), dtype=np.float64)
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    for episode in range(EPISODES):
        epsilon = max(MIN_EPSILON, EPSILON * (EPSILON_DECAY ** episode))
        episode_data = train_df.sample(frac=1, random_state=SEED + episode)
        for _, row in episode_data.iterrows():
            state_index = state_to_index[row["state"]]
            if np.random.random() < epsilon:
                action_idx = int(np.random.randint(len(ACTIONS)))
            else:
                q_values = q_table[state_index]
                best = np.flatnonzero(q_values == q_values.max())
                action_idx = int(np.random.choice(best))
            action = ACTIONS[action_idx]
            reward = reward_for(row, action)
            current = q_table[state_index, action_to_index[action]]
            # Each inspection record is terminal, as in the notebook.
            q_table[state_index, action_to_index[action]] = current + ALPHA * (reward - current)
    return states, q_table, len(train_df), len(heldout_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the team approach's score CSV and Q-learning policy. "
            "Use an existing Step 1 CSV or derive it directly from the MVTec dataset."
        )
    )
    parser.add_argument(
        "--mvtec-root", type=Path, default=Path("data/mvtec"),
        help="MVTec root directly containing grid/, zipper/, hazelnut/, leather/, and toothbrush/.",
    )
    parser.add_argument(
        "--scores-csv", type=Path, default=None,
        help="Optional existing current_anomaly_score.csv from Step 1. If omitted, use an existing project CSV or build scores from --mvtec-root.",
    )
    parser.add_argument(
        "--vision-output", type=Path,
        default=Path("team_artifacts/outputs/vision/current_anomaly_score.csv"),
        help="Where to save the generated score CSV when building directly from MVTec.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("team_artifacts/rl/shared_q_policy.json"),
        help="Policy JSON path used by app3 (default: team_artifacts/rl/shared_q_policy.json).",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="Device used for ResNet-18 feature extraction when scores must be built.",
    )
    args = parser.parse_args()
    out_path = args.output.expanduser().resolve()

    if args.scores_csv is not None:
        csv_path = args.scores_csv.expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"The provided --scores-csv path does not exist: {csv_path}")
    else:
        project_root = Path.cwd().resolve()
        candidates = [
            project_root / "outputs" / "vision" / "current_anomaly_score.csv",
            project_root / "team_artifacts" / "outputs" / "vision" / "current_anomaly_score.csv",
        ]
        existing = next((p for p in candidates if p.is_file()), None)
        if existing is not None:
            csv_path = existing
            print(f"Using existing Step 1 score CSV: {csv_path}")
        else:
            csv_path = build_score_csv_from_mvtec(args.mvtec_root, args.vision_output, args.device)

    df = load_policy_data(csv_path)
    states, q_table, training_rows, heldout_rows = train_q_table(df)
    state_records = [
        {
            "category": category,
            "risk_level": risk,
            "q_values": [float(v) for v in q_table[idx]],
            "selected_action": ACTIONS[int(np.argmax(q_table[idx]))],
        }
        for idx, (category, risk) in enumerate(states)
    ]
    payload = {
        "project": "Adaptive Visual Inspection System with RL-Based Inspection Scheduling",
        "policy_type": "shared_q_learning",
        "vision_model": "Mahalanobis anomaly score from Step 1-style output",
        "source_csv": str(csv_path),
        "source_method_filter": "mahalanobis",
        "episodes": EPISODES,
        "learning_rate": ALPHA,
        "discount_factor": GAMMA,
        "initial_epsilon": EPSILON,
        "minimum_epsilon": MIN_EPSILON,
        "epsilon_decay": EPSILON_DECAY,
        "action_costs": ACTION_COSTS,
        "missed_defect_penalty": MISSED_DEFECT_PENALTY,
        "false_rejection_penalty": FALSE_REJECTION_PENALTY,
        "risk_margin": RISK_MARGIN,
        "records_total": int(len(df)),
        "training_records": int(training_rows),
        "heldout_records_reserved_not_used_for_updates": int(heldout_rows),
        "note": (
            "Uses the notebook's offline reward setup. Labels are used to train the policy only; "
            "video-time action selection uses category and score-derived risk level. Expert review is simulated as perfect during training."
        ),
        "actions": ACTIONS,
        "risk_levels": RISK_LEVELS,
        "states": state_records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Loaded {len(df)} Mahalanobis rows from {csv_path}")
    print(f"Trained on {training_rows} rows; reserved {heldout_rows} rows from Q updates")
    print(f"Saved policy ({len(states)} states) to {out_path}")
    print("Selected actions by state:")
    for row in state_records:
        print(f"  {row['category']:10s} {row['risk_level']:6s} -> {row['selected_action']}")


if __name__ == "__main__":
    main()
