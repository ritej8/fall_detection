import numpy as np


class CentroidTracker:
    def __init__(self, max_missed_frames=5, max_match_distance=50):
        self.next_id = 0
        self.tracks = {}
        self.max_missed_frames = max_missed_frames
        self.max_match_distance = max_match_distance

    def update(self, detections):
        if len(self.tracks) == 0:
            for det in detections:
                self._create_track(det)
            return self._get_active_tracks()

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid]["centroid"] for tid in track_ids])

        if len(detections) == 0:
            for tid in track_ids:
                self._mark_missed(tid)
            return self._get_active_tracks()

        det_array = np.array(detections)
        distance_matrix = np.linalg.norm(
            track_centroids[:, np.newaxis, :] - det_array[np.newaxis, :, :], axis=2
        )

        matched_tracks, matched_dets, matches = set(), set(), []
        flat_indices = np.argsort(distance_matrix, axis=None)
        for flat_idx in flat_indices:
            t_idx, d_idx = np.unravel_index(flat_idx, distance_matrix.shape)
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            if distance_matrix[t_idx, d_idx] > self.max_match_distance:
                continue
            matches.append((track_ids[t_idx], d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        for tid, d_idx in matches:
            self.tracks[tid]["centroid"] = tuple(det_array[d_idx])
            self.tracks[tid]["missed"] = 0

        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self._mark_missed(tid)

        for d_idx in range(len(detections)):
            if d_idx not in matched_dets:
                self._create_track(tuple(det_array[d_idx]))

        return self._get_active_tracks()

    def _create_track(self, centroid):
        self.tracks[self.next_id] = {"centroid": centroid, "missed": 0}
        self.next_id += 1

    def _mark_missed(self, tid):
        self.tracks[tid]["missed"] += 1
        if self.tracks[tid]["missed"] > self.max_missed_frames:
            del self.tracks[tid]

    def _get_active_tracks(self):
        return {tid: t["centroid"] for tid, t in self.tracks.items()}