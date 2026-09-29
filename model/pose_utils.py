import cv2
import numpy as np


SKELETON_CONNECTIONS = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def draw_skeleton(frame, keypoints_xy, connections, color=(0, 255, 0), thickness=2):
    for index_a, index_b in connections:
        x1, y1 = keypoints_xy[index_a]
        x2, y2 = keypoints_xy[index_b]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    for x, y in keypoints_xy:
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    return frame


def get_reliable_hip_height(keypoints_xy, keypoints_conf, min_confidence=0.5):
    left_hip_conf = keypoints_conf[11]
    right_hip_conf = keypoints_conf[12]
    if left_hip_conf >= min_confidence and right_hip_conf >= min_confidence:
        left_hip_y = keypoints_xy[11][1]
        right_hip_y = keypoints_xy[12][1]
        return (left_hip_y + right_hip_y) / 2
    return None


def extract_reliable_hip_sequence(pose_sequence, conf_sequence, min_confidence=0.5):
    hip_heights = []
    last_reliable = None
    for pose, conf in zip(pose_sequence, conf_sequence):
        candidate = get_reliable_hip_height(pose, conf, min_confidence)
        if candidate is not None:
            last_reliable = candidate
            break
    for pose, conf in zip(pose_sequence, conf_sequence):
        hip_height = get_reliable_hip_height(pose, conf, min_confidence)
        if hip_height is not None:
            last_reliable = hip_height
        else:
            hip_height = last_reliable
        hip_heights.append(hip_height)
    return np.array(hip_heights)


def extract_hip_x(pose_sequence):
    hip_x_positions = []
    for pose in pose_sequence:
        left_hip_x = pose[11][0]
        right_hip_x = pose[12][0]
        hip_x_positions.append((left_hip_x + right_hip_x) / 2)
    return np.array(hip_x_positions)


def get_body_scale(keypoints_xy):
    shoulder_mid_x = (keypoints_xy[5][0] + keypoints_xy[6][0]) / 2
    shoulder_mid_y = (keypoints_xy[5][1] + keypoints_xy[6][1]) / 2
    hip_mid_x = (keypoints_xy[11][0] + keypoints_xy[12][0]) / 2
    hip_mid_y = (keypoints_xy[11][1] + keypoints_xy[12][1]) / 2
    dx = hip_mid_x - shoulder_mid_x
    dy = hip_mid_y - shoulder_mid_y
    return np.sqrt(dx**2 + dy**2)