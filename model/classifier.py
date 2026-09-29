import numpy as np
from model.pose_utils import extract_reliable_hip_sequence, extract_hip_x
from model.signal_processing import smooth_signal_trailing, compute_velocity, get_normalized_velocity


def is_sustained_fall_v2(velocities, i, falling_threshold=200, window_size=5, min_required=4):
    """Batch version -- checks a forward-looking window starting at i.
    Used for evaluation on complete, pre-recorded videos."""
    window = velocities[i : i + window_size]
    if len(window) < window_size:
        return False
    count_above_threshold = sum(1 for v in window if abs(v) > falling_threshold)
    return count_above_threshold >= min_required


def is_sustained_fall_live(velocities, falling_threshold=200, window_size=5, min_required=4):
    """Live-safe version -- always checks the LAST window_size values,
    never looking ahead. Used for real-time / frame-by-frame operation."""
    recent_window = velocities[-window_size:]
    if len(recent_window) < window_size:
        return False
    count_above_threshold = sum(1 for v in recent_window if abs(v) > falling_threshold)
    return count_above_threshold >= min_required


def classify_with_walking_live(pose_sequence, conf_sequence, fps,
                                 standing_threshold=0.5,
                                 falling_threshold=2.0,
                                 walking_threshold=0.3,
                                 window_size=5, min_required=4):
    """
    Returns ONE label for the current moment (the most recent frame),
    based only on data available so far -- meant to be called incrementally,
    once per new frame, with pose_sequence growing each time.
    """
    hip_x = extract_hip_x(pose_sequence)
    hip_y = extract_reliable_hip_sequence(pose_sequence, conf_sequence)

    smooth_x = smooth_signal_trailing(hip_x, window_size)
    smooth_y = smooth_signal_trailing(hip_y, window_size)

    raw_horizontal = compute_velocity(smooth_x, fps)
    raw_vertical = compute_velocity(smooth_y, fps)

    horizontal_velocity = np.array([
        get_normalized_velocity(raw_horizontal, pose_sequence, i) or 0
        for i in range(len(raw_horizontal))
    ])
    vertical_velocity = np.array([
        get_normalized_velocity(raw_vertical, pose_sequence, i) or 0
        for i in range(len(raw_vertical))
    ])

    if is_sustained_fall_live(vertical_velocity, falling_threshold, window_size, min_required):
        return "falling"

    recent_horizontal = horizontal_velocity[-window_size:]
    walking_count = sum(abs(v) > walking_threshold for v in recent_horizontal)
    if walking_count >= min_required:
        return "walking"

    recent_vertical = vertical_velocity[-window_size:]
    mean_vertical_magnitude = np.mean([abs(v) for v in recent_vertical]) if len(recent_vertical) > 0 else 0
    return "standing" if mean_vertical_magnitude < standing_threshold else "sitting"


def classify_with_walking_normalized(pose_sequence, conf_sequence, fps,
                                       standing_threshold=0.5,
                                       falling_threshold=2.0,
                                       walking_threshold=0.3,
                                       window_size=5, min_required=4):
    """
    Batch version -- returns a label for EVERY frame in the sequence.
    Used for evaluation on complete, pre-recorded videos.
    """
    hip_x = extract_hip_x(pose_sequence)
    hip_y = extract_reliable_hip_sequence(pose_sequence, conf_sequence)

    smooth_x = smooth_signal_trailing(hip_x, window_size)
    smooth_y = smooth_signal_trailing(hip_y, window_size)

    raw_horizontal = compute_velocity(smooth_x, fps)
    raw_vertical = compute_velocity(smooth_y, fps)

    horizontal_velocity = np.array([
        get_normalized_velocity(raw_horizontal, pose_sequence, i) or 0
        for i in range(len(raw_horizontal))
    ])
    vertical_velocity = np.array([
        get_normalized_velocity(raw_vertical, pose_sequence, i) or 0
        for i in range(len(raw_vertical))
    ])

    labels = []
    for i in range(len(vertical_velocity)):
        if is_sustained_fall_v2(vertical_velocity, i, falling_threshold, window_size, min_required):
            label = "falling"
        else:
            start = max(0, i - window_size + 1)
            recent_horizontal = horizontal_velocity[start:i + 1]
            walking_count = sum(abs(v) > walking_threshold for v in recent_horizontal)
            if walking_count >= min_required:
                label = "walking"
            else:
                recent_vertical = vertical_velocity[start:i + 1]
                mean_vertical_magnitude = np.mean([abs(v) for v in recent_vertical])
                label = "standing" if mean_vertical_magnitude < standing_threshold else "sitting"
        labels.append(label)

    return labels