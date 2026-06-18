#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
from pathlib import Path
import json
import numpy as np
import torch
from scipy.signal import savgol_filter, butter, filtfilt
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("Error: 'lerobot' library not found. Please install it to use this script with real robot datasets.")
    LeRobotDataset = None

def apply_savgol_filter(data, window_length=31, polyorder=3):
    """
    Applies Savitzky-Golay filter.
    window_length: Must be odd. Larger = smoother.
    polyorder: The degree of the fitted polynomial.
    """
    if len(data) < 3:
        return data
    if window_length >= len(data):
        window_length = len(data) if len(data) % 2 != 0 else len(data) - 1
    if window_length < 3:
        return data
    if window_length % 2 == 0:
        window_length -= 1
    return savgol_filter(data, window_length, polyorder)

def apply_zerophase_lowpass(data, cutoff_freq=2.0, order=2, fs=50):
    """
    Applies a zero-phase Butterworth filter (filtfilt).
    cutoff_freq: Cutoff frequency in Hz.
    fs: Sampling frequency in Hz.
    """
    if len(data) < 5:  # filtfilt requires a minimum data length depending on order
        return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    if normal_cutoff >= 1.0 or normal_cutoff <= 0.0:
        return data
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detect_staircase(data, threshold=1e-6):
    """
    Detects staircase effect (stiction/quantization).
    Returns a score representing the percentage of non-zero movements that are 'jumps'.
    """
    deltas = np.diff(data)
    abs_deltas = np.abs(deltas)
    
    moving = abs_deltas > threshold
    if not np.any(moving):
        return 0.0
    
    zero_velocity_count = np.sum(abs_deltas < threshold)
    staircase_score = zero_velocity_count / len(deltas)
    
    return staircase_score

def detect_jitter(data):
    """
    Detects jitter (high frequency noise).
    Returns the standard deviation of the second derivative (acceleration jitter).
    """
    if len(data) < 3:
        return 0.0
    accel = np.diff(data, n=2)
    return float(np.std(accel))

def parse_episode_list(ep_str, total_episodes=None):
    """
    Parses strings like '1,2,5-10,12' or 'all' into a list of integers.
    """
    if not ep_str or ep_str.lower() in ['none', '']:
        return []
    if ep_str.lower() == 'all':
        if total_episodes is not None:
            return list(range(total_episodes))
        return 'all'
    
    episodes = set()
    for part in ep_str.split(','):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                episodes.update(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                episodes.add(int(part))
            except ValueError:
                pass
    return sorted(list(episodes))

def get_episode_task(dataset, ep_idx):
    """
    Safely retrieves the task description string for a given episode index.
    """
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        episodes = dataset.meta.episodes
        if "task" in episodes:
            return episodes["task"][ep_idx]
        if "tasks" in episodes:
            val = episodes["tasks"][ep_idx]
            if isinstance(val, list) and len(val) > 0:
                return val[0]
            return str(val)
    if "task" in dataset.hf_dataset.features:
        from_idx = int(dataset.meta.episodes["dataset_from_index"][ep_idx])
        return dataset.hf_dataset[from_idx]["task"]
    return "Robot Demonstration Task"

def copy_gr00t_metadata(src_dir, dst_dir):
    """Copies GR00T specific metadata files like modality.json."""
    src_modality = Path(src_dir) / "meta" / "modality.json"
    dst_modality = Path(dst_dir) / "meta" / "modality.json"
    if src_modality.exists():
        dst_modality.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_modality, dst_modality)
        print(f"✅ Successfully copied GR00T modality.json to {dst_modality}")

def process_dataset(args):
    if LeRobotDataset is None:
        raise RuntimeError("LeRobot library must be installed to process real robot datasets.")
    
    print(f"📦 Loading source dataset: {args.repo_id} from {args.root}")
    src_dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    total_src_episodes = src_dataset.num_episodes
    print(f"ℹ️ Total episodes found in source: {total_src_episodes}")
    
    # Parse drop and filter episode lists
    drop_list = parse_episode_list(args.drop_episodes, total_src_episodes)
    filter_list = parse_episode_list(args.filter_episodes, total_src_episodes)
    
    if filter_list == 'all':
        filter_list = list(range(total_src_episodes))
        
    # Filter keys parsing
    filter_keys = [k.strip() for k in args.filter_keys.split(',') if k.strip()]
    
    print(f"Pruning/Dropping episodes: {drop_list}")
    print(f"Filtering keys {filter_keys} for episodes: {filter_list if len(filter_list) < 20 else f'{len(filter_list)} episodes'}")
    
    # Setup features and metadata for output dataset
    features = src_dataset.features
    fps = getattr(src_dataset, "fps", 30)
    use_videos = getattr(src_dataset, "use_videos", True)
    
    print(f"🔨 Creating target dataset: {args.out_repo_id} at {args.root}")
    # Remove existing target if it exists to avoid blending data corruption
    target_path = Path(args.root) / args.out_repo_id
    if target_path.exists():
        print(f"⚠️ Target directory {target_path} already exists. Overwriting...")
        shutil.rmtree(target_path)
        
    dst_dataset = LeRobotDataset.create(
        repo_id=args.out_repo_id,
        fps=fps,
        features=features,
        root=args.root,
        use_videos=use_videos
    )
    
    # --- Part 1: Process Primary Dataset Episodes ---
    processed_count = 0
    for ep_idx in tqdm(range(total_src_episodes), desc="Processing Source Dataset"):
        if ep_idx in drop_list:
            continue
            
        from_idx = int(src_dataset.meta.episodes["dataset_from_index"][ep_idx])
        to_idx = int(src_dataset.meta.episodes["dataset_to_index"][ep_idx])
        task_str = get_episode_task(src_dataset, ep_idx)
        
        # Load all frames for the episode
        ep_frames = [src_dataset[i] for i in range(from_idx, to_idx)]
        
        # Apply filtering bounded per episode if selected
        if ep_idx in filter_list:
            for key in filter_keys:
                if len(ep_frames) > 0 and key in ep_frames[0]:
                    # Convert frames sequence to a single numpy array per dimension
                    seq = np.array([f[key].cpu().numpy() for f in ep_frames])
                    filtered_seq = np.zeros_like(seq)
                    
                    if seq.ndim == 1:
                        if args.filter_type == 'savgol':
                            filtered_seq = apply_savgol_filter(seq, args.window_length, args.polyorder)
                        elif args.filter_type == 'butterworth':
                            filtered_seq = apply_zerophase_lowpass(seq, args.cutoff_freq, fs=args.fs)
                    elif seq.ndim == 2:
                        for d in range(seq.shape[1]):
                            if args.filter_type == 'savgol':
                                filtered_seq[:, d] = apply_savgol_filter(seq[:, d], args.window_length, args.polyorder)
                            elif args.filter_type == 'butterworth':
                                filtered_seq[:, d] = apply_zerophase_lowpass(seq[:, d], args.cutoff_freq, fs=args.fs)
                                
                    # Update the frames with filtered values
                    for idx, f in enumerate(ep_frames):
                        f[key] = torch.tensor(filtered_seq[idx], dtype=ep_frames[idx][key].dtype, device=ep_frames[idx][key].device)
                        
        # Append frames to target dataset
        for f in ep_frames:
            cleaned_frame = {k: v for k, v in f.items() if k in features}
            dst_dataset.add_frame(cleaned_frame)
            
        dst_dataset.save_episode(task=task_str)
        processed_count += 1
        
    print(f"✅ Successfully transferred and processed {processed_count} episodes from primary dataset.")
    
    # --- Part 2: Process Additional Dataset Episodes (if requested) ---
    if args.add_repo_id:
        print(f"📦 Loading secondary/additional dataset: {args.add_repo_id}")
        add_dataset = LeRobotDataset(repo_id=args.add_repo_id, root=args.root)
        total_add_episodes = add_dataset.num_episodes
        add_list = parse_episode_list(args.add_episodes, total_add_episodes)
        if add_list == 'all':
            add_list = list(range(total_add_episodes))
            
        print(f"Appending episodes {add_list} from secondary dataset...")
        for ep_idx in tqdm(add_list, desc="Appending Additional Episodes"):
            from_idx = int(add_dataset.meta.episodes["dataset_from_index"][ep_idx])
            to_idx = int(add_dataset.meta.episodes["dataset_to_index"][ep_idx])
            task_str = get_episode_task(add_dataset, ep_idx)
            
            ep_frames = [add_dataset[i] for i in range(from_idx, to_idx)]
            
            for f in ep_frames:
                cleaned_frame = {k: v for k, v in f.items() if k in features}
                dst_dataset.add_frame(cleaned_frame)
                
            dst_dataset.save_episode(task=task_str)
            processed_count += 1

    # --- Part 3: Consolidate and Finalize ---
    print("Consolidating new dataset and computing global statistics...")
    if hasattr(dst_dataset, "finalize"):
        dst_dataset.finalize()
    elif hasattr(dst_dataset, "consolidate"):
        dst_dataset.consolidate()
        
    # Copy GR00T-specific modality file
    src_dir = Path(args.root) / args.repo_id
    dst_dir = Path(args.root) / args.out_repo_id
    copy_gr00t_metadata(src_dir, dst_dir)
    
    print(f"🎉 Complete! Your new cleaned dataset is ready at: {dst_dir}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Interactive Dataset Builder and Time-Series Filter CLI.")
    parser.add_argument("--repo-id", type=str, required=True, help="Primary source LeRobot dataset repo ID.")
    parser.add_argument("--root", type=str, default="../data", help="Local data root directory path.")
    parser.add_argument("--out-repo-id", type=str, required=True, help="Output target LeRobot dataset repo ID.")
    
    # Dataset Editing Options
    parser.add_argument("--filter-episodes", type=str, default="none", help="Episodes to filter. e.g. '1,2,5-10' or 'all' or 'none'.")
    parser.add_argument("--drop-episodes", type=str, default="none", help="Episodes to prune/drop from the dataset entirely. e.g. '12,14,20-25'.")
    parser.add_argument("--filter-keys", type=str, default="action", help="Comma-separated keys to apply filtering on. e.g. 'action,observation.state'.")
    
    # Additional Dataset Concatenation / Blending
    parser.add_argument("--add-repo-id", type=str, default=None, help="Secondary source LeRobot dataset repo ID to append episodes from.")
    parser.add_argument("--add-episodes", type=str, default="all", help="Specific episodes from secondary dataset to append. e.g. '0-5' or 'all'.")
    
    # Filter Parameter Tuning
    parser.add_argument("--filter-type", type=str, choices=['savgol', 'butterworth'], default='savgol', help="Time-series filter algorithm.")
    parser.add_argument("--window-length", type=int, default=31, help="SavGol window length filter parameter (must be odd).")
    parser.add_argument("--polyorder", type=int, default=3, help="SavGol polynomial degree parameter.")
    parser.add_argument("--cutoff-freq", type=float, default=2.0, help="Butterworth lowpass cutoff frequency parameter in Hz.")
    parser.add_argument("--fs", type=float, default=50.0, help="Sampling frequency framework specification in Hz.")

    args = parser.parse_args()
    process_dataset(args)

if __name__ == "__main__":
    main()
