#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive Dataset Validation & Stationarity Analysis for LeRobot.

Phase 1 (Within-Episode): Checks action and observation increments (deltas) 
for difference-stationarity to ensure hardware signal smoothness.
Phase 2 (Across-Episode): Groups episodes chronologically to test for expert 
strategy drift, fatigue, and environment initialization bias using Kruskal-Wallis.

DEPENDENCIES
------------
pip install lerobot statsmodels pandas numpy scipy tqdm

Usage:
    python scripts/validate_lerobot_dataset.py --repo-id <repo_id> --root <data_dir> [options]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from tqdm import tqdm

# Add root to sys.path to allow imports if needed
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Error: 'lerobot' not found. Please install it with 'pip install lerobot'.")
    sys.exit(1)

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except ImportError:
    print("Error: 'statsmodels' not found. Please install it with 'pip install statsmodels'.")
    sys.exit(1)


def classify(adf_p: float, kpss_p: float, alpha: float) -> str:
    """Joint verdict from ADF + KPSS p-values."""
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p >= alpha
    if adf_stationary and kpss_stationary:
        return "stationary"
    if (not adf_stationary) and (not kpss_stationary):
        return "non_stationary"
    if adf_stationary and (not kpss_stationary):
        return "difference_stationary"
    return "trend_stationary"


def safe_adf(x: np.ndarray, max_lag: int | None, autolag: str | None) -> float:
    if np.allclose(x, x[0]):
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(adfuller(x, maxlag=max_lag, autolag=autolag)[1])
    except Exception:
        return np.nan


def safe_kpss(x: np.ndarray, nlags: str | int) -> float:
    if np.allclose(x, x[0]):
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(x, regression="c", nlags=nlags)[1])
    except Exception:
        return np.nan


def get_episode_matrix(dataset: LeRobotDataset, ep_idx: int, key: str) -> np.ndarray | None:
    """Extracts a specific feature array (e.g., 'action') for a given episode."""
    if key not in dataset.hf_dataset.features:
        return None
    from_idx = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
    to_idx = int(dataset.meta.episodes["dataset_to_index"][ep_idx])
    data = dataset.hf_dataset.select(range(from_idx, to_idx))[key]
    return np.asarray(data, dtype=np.float64)


def test_across_episode_drift(
    metrics_df: pd.DataFrame, num_buckets: int, alpha: float
) -> pd.DataFrame:
    """Runs Kruskal-Wallis tests across chronological buckets to detect drift."""
    metrics_df = metrics_df.sort_values("episode").reset_index(drop=True)
    metrics_df["bucket"] = pd.qcut(metrics_df.index, q=num_buckets, labels=False, duplicates="drop")
    
    drift_results = []
    
    # Test all numeric columns dynamically (excluding episode/bucket IDs)
    cols_to_test = [c for c in metrics_df.columns if c not in ["episode", "bucket"]]
    
    for col in cols_to_test:
        groups = [group[col].values for _, group in metrics_df.groupby("bucket")]
        if len(groups) > 1 and all(len(g) > 0 for g in groups):
            try:
                stat, p_val = kruskal(*groups)
                drift_results.append({
                    "metric": col,
                    "p_value": p_val,
                    "drift_detected": p_val < alpha
                })
            except ValueError:
                pass # All values identical or invalid for Kruskal

    return pd.DataFrame(drift_results)


def main():
    parser = argparse.ArgumentParser(description="Validate LeRobot dataset stationarity and drift.")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo ID.")
    parser.add_argument("--root", type=Path, required=True, help="Local root directory for datasets.")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Specific episodes to test.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--min-len", type=int, default=20, help="Minimum episode length.")
    parser.add_argument("--num-buckets", type=int, default=3, help="Number of chronological buckets for drift testing.")
    parser.add_argument("--out-dir", type=Path, default=Path("./dataset_validation_out"), help="Output directory.")
    parser.add_argument("--diff-order", type=int, default=1, help="Differencing order (0=raw, 1=velocity, 2=accel).")
    parser.add_argument("--adf-autolag", type=str, default="AIC", help="ADF autolag method.")
    parser.add_argument("--max-lag", type=int, default=None, help="ADF maximum lag.")
    parser.add_argument("--kpss-nlags", type=str, default="auto", help="KPSS number of lags.")
    
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading dataset repo_id={args.repo_id!r} from root={args.root}")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    
    episodes = args.episodes if args.episodes is not None else list(range(dataset.num_episodes))
    
    within_ep_rows: list[dict] = []
    across_ep_metrics: list[dict] = []

    for ep in tqdm(episodes, desc="Validating Episodes"):
        # 1. Fetch Data
        A_action = get_episode_matrix(dataset, ep, "action")
        A_obs = get_episode_matrix(dataset, ep, "observation.state")
        
        if A_action is None or A_action.ndim != 2 or A_action.shape[0] < args.min_len:
            continue

        # 2. Phase 2 Metrics Setup (Across-Episode)
        ep_metrics = {
            "episode": ep,
            "trajectory_length": A_action.shape[0],
        }

        # Check Environment Bias: Did the initial starting state drift over time?
        if A_obs is not None and A_obs.shape[0] > 0:
            initial_state = A_obs[0, :]
            for d in range(initial_state.shape[0]):
                ep_metrics[f"init_obs_dim_{d}"] = initial_state[d]

        # 3. Analyze Data Streams (Phase 1 & variation metrics)
        data_streams = {"action": A_action}
        if A_obs is not None:
            data_streams["observation.state"] = A_obs

        for stream_name, raw_matrix in data_streams.items():
            # Apply differencing order (d=1 for deltas, d=0 for raw)
            if args.diff_order > 0:
                data_to_test = np.diff(raw_matrix, n=args.diff_order, axis=0)
            else:
                data_to_test = raw_matrix
            
            for d in range(data_to_test.shape[1]):
                x_series = data_to_test[:, d]
                
                # Phase 2: Record total variation to check for fatigue/laziness
                ep_metrics[f"var_{stream_name}_dim_{d}"] = np.sum(np.abs(x_series))
                
                # Phase 1: Stationarity of the signal
                adf_p = safe_adf(x_series, args.max_lag, args.adf_autolag)
                kpss_p = safe_kpss(x_series, args.kpss_nlags)
                verdict = (
                    "constant" if (np.isnan(adf_p) and np.isnan(kpss_p))
                    else classify(
                        adf_p if not np.isnan(adf_p) else 1.0,
                        kpss_p if not np.isnan(kpss_p) else 0.0,
                        args.alpha,
                    )
                )
                within_ep_rows.append({
                    "episode": ep,
                    "stream": stream_name,
                    "dim": d,
                    "adf_p": adf_p,
                    "kpss_p": kpss_p,
                    "verdict": verdict,
                })
                
        across_ep_metrics.append(ep_metrics)

    if not within_ep_rows:
        print("No episodes validated. Check if the dataset has 'action' features and meets the minimum length.")
        return

    # --- Save and Report Phase 1 ---
    df_within = pd.DataFrame(within_ep_rows)
    within_path = args.out_dir / f"within_episode_d{args.diff_order}.csv"
    df_within.to_csv(within_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"PHASE 1: WITHIN-EPISODE SMOOTHNESS (Differencing d={args.diff_order})")
    print(f"{'='*60}")
    
    # Group summaries by stream (actions vs observations)
    summary = df_within.groupby(["stream", "verdict"]).size().unstack(fill_value=0)
    summary_pct = summary.div(summary.sum(axis=1), axis=0).round(3) * 100
    print("Percentage of Dimensions by Verdict:")
    print(summary_pct.to_string())
    
    # --- Save and Report Phase 2 ---
    df_metrics = pd.DataFrame(across_ep_metrics)
    metrics_path = args.out_dir / "episode_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    
    df_drift = test_across_episode_drift(df_metrics, num_buckets=args.num_buckets, alpha=args.alpha)
    drift_path = args.out_dir / "across_episode_drift.csv"
    df_drift.to_csv(drift_path, index=False)

    print(f"✅ Saved metrics to {metrics_path}")
    print(f"✅ Saved drift results to {drift_path}")

    print(f"\n{'='*60}")
    print(f"PHASE 2: ACROSS-EPISODE DRIFT (Chronological Buckets: {args.num_buckets})")
    print(f"{'='*60}")
    
    if not df_drift.empty:
        drift_detected = df_drift[df_drift["drift_detected"] == True]
        if drift_detected.empty:
            print("✅ No significant strategy or initialization drift detected.")
        else:
            print("⚠️ WARNING: Expert or Environment drift detected in:")
            # Sort by p-value to show the most severe drift first
            drift_detected = drift_detected.sort_values("p_value")
            print(drift_detected[["metric", "p_value"]].to_string(index=False))
            print("\nConsider bucketing your dataset or removing fatiguing episodes.")
    else:
        print("Not enough variance to calculate drift.")


if __name__ == "__main__":
    main()
