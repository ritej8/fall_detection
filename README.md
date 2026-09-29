# Human Activity Detection & Fall Detection

A computer vision pipeline that detects, tracks, and classifies human activity
(standing, sitting, walking, falling) from video, using pose estimation and
rule-based temporal analysis — with a documented evaluation against real,
labeled data and an honest account of what works and what doesn't.

---

## Architecture

```
Video
  ↓
Person Detection + Pose Estimation (YOLO11-pose)
  ↓
Multi-Person Tracking (custom centroid tracker with data association)
  ↓
Confidence-Aware Pose Filtering
  ↓
Body-Scale Normalization (torso-length reference)
  ↓
Temporal Feature Extraction (vertical & horizontal velocity)
  ↓
Rule-Based Activity Classification (standing / sitting / walking / falling)
  ↓
Event Logging & Per-Person Summaries
  ↓
Annotated Output Video
```

---

## What this system does

- Detects and tracks multiple people simultaneously in a video, assigning
  persistent IDs across frames
- Extracts 17-point body pose per person, per frame (YOLO11-pose)
- Filters out low-confidence (occluded/uncertain) keypoints rather than
  trusting every detection blindly
- Normalizes movement measurements by each person's own body scale, so
  the system isn't dependent on camera distance or video resolution
- Classifies each person's activity using velocity-based rules, with
  sustained-motion checks to avoid single-frame false alarms
- Produces a structured event log per person: what happened, when, for
  how long, and whether a fall was detected

---

## Evaluation

Evaluated against the public **Human Action Recognition** dataset (Kaggle,
7 labeled classes). Evaluation scope was deliberately limited to the 4
classes this system directly targets (Standing, Sitting, Walking, Fall_down);
Sit_down, Stand_up, and Lying_down were excluded as they don't map cleanly
onto this system's output categories — a documented scope decision, not an
oversight.

**Sample: 19 videos (5 per class, 1 skipped for insufficient pose data)**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Standing | 0.33 | 0.25 | 0.29 |
| Sitting | 0.50 | 0.20 | 0.29 |
| Walking | 0.50 | 0.60 | 0.55 |
| Falling | 0.50 | **0.80** | 0.62 |

**Key finding:** fall detection recall (0.80) is the strongest metric —
appropriate for a safety-oriented system, where missing a real fall is worse
than a false alarm. Precision is lower across all classes, driven mainly by
confusion between Standing/Walking and Standing/Sitting.

---

## Real engineering investigation, not just a working demo

This project involved genuine debugging against real data, not just building
something that worked on one clean example:

1. **Discovered a false positive**: a fast sitting-adjustment in self-recorded
   footage was misclassified as a fall (raw velocity spiked to 1040 px/s vs.
   the real fall's 549 px/s in the same clip). Fixed by requiring sustained
   elevated velocity across multiple frames, not a single-frame spike.

2. **Discovered a false negative**: a gentler real fall (from the public
   dataset) was missed because the sustained-detection fix was initially too
   strict (required ALL consecutive frames above threshold; one brief dip
   broke the detection). Fixed by requiring a majority of frames in a window,
   not all of them.

3. **Discovered a generalization failure**: raw pixel-based velocity
   thresholds, calibrated on one video, completely failed to generalize —
   0/5 real falls detected on a broader sample. Diagnosed as a camera-distance
   scale problem and fixed by normalizing velocity against each person's own
   body size (torso length), improving fall recall from 0/5 to 4/5 on the
   same test videos.

4. **Discovered a live-vs-batch mismatch**: the classifier worked correctly
   when evaluated on complete, pre-recorded videos, but failed to detect the
   same real fall when run frame-by-frame in the live pipeline (as a real
   deployment would). Root cause: two components silently depended on
   "future" frames that don't exist yet in a live setting — a centered
   smoothing window and a forward-scanning sustained-fall check. Fixed by
   replacing both with strictly backward-looking (causal) versions, verified
   by confirming identical output whether the classifier saw the full video
   or only the frames available up to that point in time.

---

## Runtime Performance

Measured on Kaggle's standard notebook environment (excluding model warm-up,
steady-state timing only — same methodology as the Week 2 detection
benchmark this project builds on):

| Metric | Value |
|---|---|
| Source video frame rate | 30 FPS |
| Achievable pipeline FPS | 12.4 FPS |
| Real-time capable? | No (≈2.4x too slow) |

The bottleneck is pose model inference itself, not the tracking or
classification logic added on top (consistent with the same finding from
the Week 2 detection-only benchmark on this hardware). Standard fixes for
real-time deployment — GPU acceleration, a smaller pose model variant, or
frame skipping — were not applied here but are well-understood, previously
demonstrated techniques (see Week 2 frame-skipping experiments) that would
directly close this gap.

## False Positive / False Negative Analysis

Beyond the aggregate precision/recall numbers, three specific, diagnosed
failure patterns were found during development:

**False positive — brief non-fall movement misread as a fall.** A fast
sitting-adjustment in self-recorded footage produced a raw velocity spike
(1040 px/s) exceeding a real fall's peak (549 px/s) in the same clip. Root
cause: a single-frame velocity spike, with no requirement that the motion
sustain over time. Fixed by requiring elevated velocity across multiple
consecutive frames before confirming a fall.

**False negative — a real, gentler fall went undetected.** A real
`Fall_down`-labeled video's peak velocity (215 px/s) narrowly and
inconsistently crossed the fall threshold, with one frame dipping just below
it mid-event. The initial "sustained fall" fix required ALL frames in a
window to exceed the threshold, so this single dip caused the whole
detection to fail. Fixed by requiring a majority (4 of 5) of frames in the
window to exceed the threshold, rather than all of them.

**Systemic false negatives — fall detection failed across a broader,
independent sample (0/5 correct).** Root cause: fixed pixel-velocity
thresholds, calibrated on one video's camera distance/resolution, do not
generalize to videos filmed differently. Confirmed directly: the same real
fall produced very different raw velocity magnitudes depending on how close
the camera was to the subject. Fixed by normalizing velocity against each
person's own body scale (shoulder-to-hip distance), which is invariant to
camera distance. This improved fall recall on the same test set from 0/5 to
4/5.

Together, these three findings — captured through direct measurement rather
than assumption — account for the residual 0.50 precision on the Falling
class reported in the evaluation table: the system still occasionally
over-triggers on other classes (2 Sitting and 2 Walking videos were
misclassified as Falling in the 19-video sample), a known trade-off of
prioritizing high recall for a safety-relevant detection task.

## Known limitations

- **Rule-based thresholds, not learned**: velocity-based rules work well for
  clearly falling vs. clearly standing, but struggle with intermediate cases
  (sitting down quickly, standing still with natural sway). A learned
  temporal model (e.g., LSTM or 1D-CNN over pose sequences) is the natural
  next step — deferred deliberately, not attempted here.
- **Single-frame fall trigger**: the system flags the *onset* of a fall
  rather than labeling the fall's full duration. Appropriate for triggering
  an alert, not for frame-accurate temporal segmentation.
- **Standing/Sitting overlap**: these two states have genuinely overlapping
  velocity signatures in this system; a single threshold cannot fully
  separate them (confirmed via direct measurement, not assumed).
- **Small evaluation sample**: 19 videos is a starting point, not a
  statistically robust benchmark. A full evaluation would use the complete
  dataset.

## Bugs found and fixed during development (beyond the classifier logic)

- **Multi-person data-association bug**: the pipeline's frame-to-track
  matching logic could assign the same detected person's pose to two
  different tracked IDs simultaneously when multiple people were present,
  since it always searched the full detection list rather than excluding
  already-claimed detections. Did not affect any single-person evaluation
  video (the only case tested so far), but was identified and fixed before
  any multi-person deployment, since it would otherwise scramble tracking on
  first use with more than one person.

---

## What a V2 would address

- Replace rule-based classification with a learned model trained on pose
  sequences, to handle the Standing/Sitting/Walking overlap that fixed
  thresholds cannot resolve
- Full-dataset evaluation (all videos per class, not a 5-video sample)
- Per-frame (not per-video) evaluation for finer-grained accuracy measurement
- Address the single-frame trigger limitation if full-duration event
  labeling becomes a requirement

---

## Tech stack

- **Pose estimation**: YOLO11-pose (Ultralytics)
- **Tracking**: custom centroid tracker (distance-based data association,
  temporal stability via missed-frame tolerance)
- **Analysis**: NumPy-based velocity/normalization/classification pipeline
- **Evaluation**: custom confusion matrix / precision / recall / F1
  implementation
## API

A minimal REST API (FastAPI) exposes the pipeline for real use.

### Running the API

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive
documentation at `http://127.0.0.1:8000/docs`.

### Endpoint: `POST /analyze`

Accepts a video file upload, runs the full pipeline (detection, tracking,
classification), and returns a structured JSON summary.

**Example response:**
```json
{
  "video_processed": true,
  "filename": "video_263.avi",
  "fps": 30,
  "total_frames": 36,
  "persons_detected": 1,
  "results": [
    {
      "track_id": 0,
      "events": [
        {"activity": "sitting", "start_time_sec": 0, "duration_sec": 0.13},
        {"activity": "falling", "start_time_sec": 0.13, "duration_sec": 0.23}
      ],
      "total_durations": {"sitting": 0.13, "falling": 0.23},
      "fell": true
    }
  ]
}
```

### Not included (deferred, documented as future work)
- Docker containerization
- Cloud deployment
- Authentication / rate limiting
- Concurrent request handling