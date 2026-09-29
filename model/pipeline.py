import cv2
import numpy as np

from model.tracker import CentroidTracker
from model.pose_utils import draw_skeleton, SKELETON_CONNECTIONS
from model.classifier import classify_with_walking_live
from model.events import build_person_summary


def run_pipeline(video_path, output_path, model, min_frames_before_classifying=5):
    """
    Runs the full pipeline on a video file:
    pose detection, tracking, confidence-aware filtering,
    live activity classification, and annotated video output.

    Returns:
        pose_history_per_id
        conf_history_per_id
        fps
        total_frames
    """

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracker = CentroidTracker(
        max_missed_frames=5,
        max_match_distance=250
    )

    pose_history_per_id = {}
    conf_history_per_id = {}

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # -----------------------------
        # 1. Pose detection
        # -----------------------------

        results = model(frame, verbose=False)

        n_people = len(results[0].keypoints.xy)



        centroids = []
        poses_this_frame = []
        confs_this_frame = []

        # -----------------------------
        # 2. Extract pose information
        # -----------------------------

        for p in range(n_people):

            kp = results[0].keypoints.xy[p].cpu().numpy()

            kp_conf = results[0].keypoints.conf[p].cpu().numpy()

            # Hip centroid
            hip_x = (kp[11][0] + kp[12][0]) / 2
            hip_y = (kp[11][1] + kp[12][1]) / 2

            centroids.append((hip_x, hip_y))

            poses_this_frame.append(kp)
            confs_this_frame.append(kp_conf)

        # -----------------------------
        # 3. Update tracker
        # -----------------------------

        active_tracks = tracker.update(centroids)

        already_matched_indices = set()

        # -----------------------------
        # 4. Match tracks to detections
        # -----------------------------

        for track_id, track_centroid in active_tracks.items():

            if not poses_this_frame:
                continue

            available_indices = [
                i
                for i in range(len(centroids))
                if i not in already_matched_indices
            ]

            if not available_indices:
                continue

            distances = [
                np.linalg.norm(
                    np.array(track_centroid)
                    - np.array(centroids[i])
                )
                for i in available_indices
            ]

            closest_idx = available_indices[
                int(np.argmin(distances))
            ]

            already_matched_indices.add(closest_idx)

            matched_pose = poses_this_frame[closest_idx]
            matched_conf = confs_this_frame[closest_idx]

            # -----------------------------
            # 5. Store pose history
            # -----------------------------

            pose_history_per_id.setdefault(
                track_id, []
            ).append(matched_pose)

            conf_history_per_id.setdefault(
                track_id, []
            ).append(matched_conf)

            pose_hist = pose_history_per_id[track_id]
            conf_hist = conf_history_per_id[track_id]

            # -----------------------------
            # 6. Activity classification
            # -----------------------------

            label = "collecting data..."

            if len(pose_hist) >= min_frames_before_classifying:

                label = classify_with_walking_live(
                    pose_hist,
                    conf_hist,
                    fps=fps
                )

            # -----------------------------
            # 7. Draw pose
            # -----------------------------

            frame = draw_skeleton(
                frame,
                matched_pose,
                SKELETON_CONNECTIONS
            )

            text_pos = (
                int(matched_pose[0][0]),
                int(matched_pose[0][1]) - 20
            )

            color = (
                (0, 0, 255)
                if label == "falling"
                else (0, 255, 0)
            )

            cv2.putText(
                frame,
                f"ID {track_id}: {label}",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # -----------------------------
        # 8. Save frame
        # -----------------------------

        out.write(frame)

        frame_idx += 1

    # -----------------------------
    # 9. Release resources
    # -----------------------------

    cap.release()
    out.release()

    print(
        f"Processed {frame_idx} frames. "
        f"Saved to {output_path}"
    )

    return (
        pose_history_per_id,
        conf_history_per_id,
        fps,
        frame_idx
    )


def run_pipeline_and_get_summary(
    video_path,
    output_path,
    model,
    min_frames_before_classifying=5
):
    """
    Runs the full pipeline and returns a structured
    JSON-ready summary for every tracked person.

    This is the function used by the API.
    """

    # -----------------------------
    # 1. Run the pipeline
    # -----------------------------

    (
        pose_history_per_id,
        conf_history_per_id,
        fps,
        total_frames
    ) = run_pipeline(
        video_path,
        output_path,
        model,
        min_frames_before_classifying
    )

    # -----------------------------
    # 2. Build person summaries
    # -----------------------------

    results = []

    for track_id, pose_hist in pose_history_per_id.items():

        conf_hist = conf_history_per_id[track_id]

        labels = []

        # We need enough frames before classification starts
        for i in range(
            min_frames_before_classifying,
            len(pose_hist) + 1
        ):

            label = classify_with_walking_live(
                pose_hist[:i],
                conf_hist[:i],
                fps=fps
            )

            labels.append(label)

        # If there are not enough observations,
        # don't try to build an activity summary.
        if not labels:
            summary = {
                "track_id": track_id,
                "events": [],
                "total_durations": {},
                "fell": False
            }

        else:
            summary = build_person_summary(
                track_id,
                labels,
                fps
            )

        results.append(summary)

    # -----------------------------
    # 4. Return API-ready result
    # -----------------------------

    return {
        "fps": fps,
        "total_frames": total_frames,
        "persons_detected": len(pose_history_per_id),
        "results": results
    }