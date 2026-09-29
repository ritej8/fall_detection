import numpy as np
from model.pose_utils import get_body_scale


def smooth_signal_trailing(signal, window_size=5):
    smoothed = []
    for i in range(len(signal)):
        start = max(0, i - window_size + 1)
        window = signal[start:i + 1]
        smoothed.append(np.mean(window))
    return np.array(smoothed)


def compute_velocity(signal, fps):
    velocities = []
    for i in range(1, len(signal)):
        velocities.append((signal[i] - signal[i - 1]) * fps)
    return np.array(velocities)


def get_normalized_velocity(velocity, pose_sequence, i):
    scale_start = get_body_scale(pose_sequence[i])
    scale_end = get_body_scale(pose_sequence[i + 1])
    body_scale = (scale_start + scale_end) / 2
    if body_scale <= 0:
        return None
    return velocity[i] / body_scale