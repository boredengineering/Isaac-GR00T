#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt

def apply_savgol_filter(data, window_length=31, polyorder=3):
    """
    Applies Savitzky-Golay filter.
    window_length: Must be odd. Larger = smoother.
    polyorder: The degree of the fitted polynomial.
    """
    # Ensure window is odd and less than data length
    if window_length >= len(data):
        window_length = len(data) if len(data) % 2 != 0 else len(data) - 1
    if window_length < 3:
        return data
    if window_length % 2 == 0:
        window_length -= 1
    return savgol_filter(data, window_length, polyorder)

def apply_zerophase_lowpass(data, cutoff_freq=0.08, order=2, fs=100):
    """
    Applies a zero-phase Butterworth filter (filtfilt).
    cutoff_freq: Cutoff frequency in Hz.
    fs: Sampling frequency in Hz.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    if normal_cutoff >= 1.0:
        return data
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detect_staircase(data, threshold=0.01):
    """
    Detects staircase effect (stiction/quantization).
    Returns a score representing the percentage of non-zero movements that are 'jumps'.
    """
    deltas = np.diff(data)
    abs_deltas = np.abs(deltas)
    
    # Movements larger than threshold
    moving = abs_deltas > 1e-6
    if not np.any(moving):
        return 0.0
    
    # A staircase has many zero deltas punctuated by large jumps
    # Score = proportion of 'moving' steps where the delta is significantly larger than the average 'smooth' delta
    # Or more simply: ratio of (steps where delta == 0) / (total steps) during an active period
    
    zero_velocity_count = np.sum(abs_deltas < 1e-6)
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
    return np.std(accel)

def simulate_steppy_data(length=500):
    """Creates synthetic data resembling staircase effect."""
    t = np.linspace(0, 10, length)
    smooth_cmd = np.sin(t) * 0.5 + 0.5
    
    # Simulate stiction/quantization (staircase effect)
    steppy_state = np.zeros_like(smooth_cmd)
    current_val = smooth_cmd[0]
    threshold = 0.08  # The hand won't move until error > 0.08
    
    for i in range(length):
        if abs(smooth_cmd[i] - current_val) > threshold:
            current_val = smooth_cmd[i]
        steppy_state[i] = current_val
        
    return t, smooth_cmd, steppy_state

def main():
    # 1. Generate synthetic data
    t, cmd, steppy_state = simulate_steppy_data()

    # 2. Apply Filters
    smoothed_savgol = apply_savgol_filter(steppy_state, window_length=41, polyorder=3)
    smoothed_butter = apply_zerophase_lowpass(steppy_state, cutoff_freq=2.0, fs=50) # Assuming 50Hz

    # 3. Analyze noise
    staircase_score = detect_staircase(steppy_state)
    jitter_score = detect_jitter(steppy_state)
    print(f"Detected Staircase Score: {staircase_score:.4f}")
    print(f"Detected Jitter Score: {jitter_score:.4f}")

    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.set_title("Position (Joint State)")
    ax1.plot(t, cmd, '--', color='gray', alpha=0.5, label='Command (Smooth)')
    ax1.plot(t, steppy_state, label='Raw Hardware State (Staircase)', linewidth=2)
    ax1.plot(t, smoothed_savgol, label='SavGol Smoothed', linewidth=2)
    ax1.plot(t, smoothed_butter, label='Zero-Phase Lowpass Smoothed', linewidth=2, linestyle=':')
    ax1.set_ylabel("Normalized Position")
    ax1.legend()
    ax1.grid(alpha=0.3)

    delta_steppy = np.gradient(steppy_state)
    delta_savgol = np.gradient(smoothed_savgol)

    ax2.set_title("Delta Action (Derivative of Position)")
    ax2.plot(t, delta_steppy, label='Raw Delta (Spiky)', color='orange', alpha=0.7)
    ax2.plot(t, delta_savgol, label='SavGol Delta (Smooth)', color='green', linewidth=2)
    ax2.set_ylabel("Change in Position")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
