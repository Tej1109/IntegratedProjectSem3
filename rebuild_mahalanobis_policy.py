from __future__ import annotations

"""Rebuild the Step 2-compatible Mahalanobis Q-policy from a Step 1 score CSV.

This trains the Q-table on the stratified training split and writes a held-out audit,
without repeating ResNet-18 feature extraction. It uses the supplied notebook's
three actions and illustrative costs/penalties.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "team_artifacts" / "outputs" / "vision" / "current_anomaly_score.csv"
RL_DIR = ROOT / "team_artifacts" / "rl"
POLICY_PATH = RL_DIR / "shared_q_policy.json"
AUDIT_PATH = RL_DIR / "heldout_policy_decisions.csv"

SEED = 42
CATEGORIES = ["grid", "zipper", "hazelnut", "leather", "toothbrush"]
VISION_METHOD = "mahalanobis"
ACTIONS = ["skip", "prediction", "expert"]
ACTION_COSTS = {"skip": 0, "prediction": 1, "expert": 5}
MISSED_DEFECT_PENALTY = 10
FALSE_REJECTION_PENALTY = 3
ALPHA = 0.1
GAMMA = 0.9
EPISODES = 500
EPSILON = 0.20
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05
RISK_MARGIN = 0.10


def reward_for(row: pd.Series, action: str) -> float:
    label = int(row["label"])
    prediction = int(row["prediction"])
    reward = -ACTION_COSTS[action]
    if action == "skip":
        missed_defect, false_rejection = label == 1, False
    elif action == "prediction":
        missed_defect = label == 1 and prediction == 0
        false_rejection = label == 0 and prediction == 1
    elif action == "expert":
        missed_defect, false_rejection = False, False
    else:
        raise ValueError(f"Unknown action: {action}")
    if missed_defect:
        reward -= MISSED_DEFECT_PENALTY
    if false_rejection:
        reward -= FALSE_REJECTION_PENALTY
    return float(reward)


def main() -> None:
    if not DATA_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_CSV}. The Colab asset bundle must include "
            "team_artifacts/outputs/vision/current_anomaly_score.csv. If it is absent, "
            "rerun colab_asset_builder.ipynb and extract MVTec_Team_Assets.zip."
        )
    df = pd.read_csv(DATA_CSV)
    required = {"category", "method", "label", "prediction", "normalized_score", "normalized_threshold", "image_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Score CSV is missing required columns: {missing}")
    rl_df = df[df["method"].astype(str).str.lower().eq(VISION_METHOD)].copy().reset_index(drop=True)
    if rl_df.empty:
        raise ValueError(f"No Mahalanobis records found in {DATA_CSV}")
    rl_df["label"] = rl_df["label"].astype(int)
    rl_df["prediction"] = rl_df["prediction"].astype(int)
    margin = rl_df["normalized_score"].astype(float) - rl_df["normalized_threshold"].astype(float)
    rl_df["risk_level"] = np.where(margin < -RISK_MARGIN, "low", np.where(margin < 0, "medium", "high"))
    rl_df["state"] = list(zip(rl_df["category"].astype(str), rl_df["risk_level"].astype(str)))
    stratify_key = rl_df["category"].astype(str) + "_" + rl_df["label"].astype(str)
    train_df, heldout_df = train_test_split(rl_df, test_size=0.20, random_state=SEED, stratify=stratify_key)
    train_df, heldout_df = train_df.reset_index(drop=True), heldout_df.reset_index(drop=True)
    states = sorted(set(rl_df["state"].tolist()))
    state_to_index = {state: i for i, state in enumerate(states)}
    q_table = np.zeros((len(states), len(ACTIONS)), dtype=np.float64)
    rng = np.random.RandomState(SEED)

    for episode in range(EPISODES):
        epsilon = max(MIN_EPSILON, EPSILON * (EPSILON_DECAY ** episode))
        shuffled = train_df.sample(frac=1, random_state=SEED + episode)
        for _, row in shuffled.iterrows():
            state_i = state_to_index[row["state"]]
            if rng.random_sample() < epsilon:
                action_i = int(rng.randint(len(ACTIONS)))
            else:
                q_vals = q_table[state_i]
                choices = np.flatnonzero(q_vals == q_vals.max())
                action_i = int(rng.choice(choices))
            action = ACTIONS[action_i]
            reward = reward_for(row, action)
            # Terminal update matches the Step 2 training loop.
            q_table[state_i, action_i] += ALPHA * (reward - q_table[state_i, action_i])

    metadata = {
        "project": "Adaptive Visual Inspection System with RL-Based Inspection Scheduling",
        "source_notebooks": ["step1_vision.ipynb", "step2_RL_Scheduler.ipynb"],
        "vision_method": VISION_METHOD,
        "actions_in_q_value_order": ACTIONS,
        "state_definition": "(category, risk_level); risk from normalized_score - normalized_threshold; margin < -0.10 low, < 0 medium, otherwise high",
        "training_protocol": "Step 2-compatible stratified 80/20 split; train split only; independent terminal cases; 500 episodes",
        "random_seed": SEED,
        "training_records": int(len(train_df)),
        "heldout_records": int(len(heldout_df)),
        "episodes": EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "initial_epsilon": EPSILON,
        "epsilon_decay": EPSILON_DECAY,
        "minimum_epsilon": MIN_EPSILON,
        "action_costs": ACTION_COSTS,
        "missed_defect_penalty": MISSED_DEFECT_PENALTY,
        "false_rejection_penalty": FALSE_REJECTION_PENALTY,
        "note": "Labels are used only for offline simulated rewards/audit. Expert is assumed perfect in the notebook simulation. This is the shared held-out-split Q-table, not the later realistic-stream or category-specific variants.",
    }
    payload = {"metadata": metadata, "states": [
        {"category": cat, "risk_level": risk, "q_values": [float(v) for v in q_table[i]]}
        for i, (cat, risk) in enumerate(states)
    ]}
    RL_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    audit_rows = []
    for _, row in heldout_df.iterrows():
        action = ACTIONS[int(np.argmax(q_table[state_to_index[row["state"]]]))]
        label, pred = int(row["label"]), int(row["prediction"])
        final_decision = 0 if action == "skip" else (pred if action == "prediction" else label)
        audit_rows.append({
            "image_path": row["image_path"], "category": row["category"], "risk_level": row["risk_level"],
            "label": label, "mahalanobis_prediction": pred, "selected_action": action,
            "final_decision": final_decision, "missed_defect": int(label == 1 and final_decision == 0),
            "false_rejection": int(label == 0 and final_decision == 1),
        })
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT_PATH, index=False)
    print(f"Saved Step 2-compatible policy: {POLICY_PATH}")
    print(f"Saved held-out audit: {AUDIT_PATH}")
    print(f"Training records: {len(train_df):,} | held-out records: {len(heldout_df):,} | states: {len(states)}")
    print("Held-out actions:")
    print(audit_df["selected_action"].value_counts().to_string())


if __name__ == "__main__":
    main()
