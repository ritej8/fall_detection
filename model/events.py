def extract_events(labels, fps):
    events = []
    if not labels:
        return events

    current_activity = labels[0]
    start_frame = 0

    for i in range(1, len(labels)):
        if labels[i] != current_activity:
            duration_frames = i - start_frame
            events.append({
                "activity": current_activity,
                "start_frame": start_frame,
                "duration_frames": duration_frames,
                "start_time_sec": start_frame / fps,
                "duration_sec": duration_frames / fps
            })
            current_activity = labels[i]
            start_frame = i

    duration_frames = len(labels) - start_frame
    events.append({
        "activity": current_activity,
        "start_frame": start_frame,
        "duration_frames": duration_frames,
        "start_time_sec": start_frame / fps,
        "duration_sec": duration_frames / fps
    })

    return events


def filter_short_events(events, min_duration_frames=3):
    if not events:
        return events
    filtered = [dict(events[0])]
    for event in events[1:]:
        event = dict(event)
        prev = filtered[-1]
        if event["duration_frames"] < min_duration_frames:
            prev["duration_frames"] += event["duration_frames"]
            prev["duration_sec"] += event["duration_sec"]
        elif event["activity"] == prev["activity"]:
            prev["duration_frames"] += event["duration_frames"]
            prev["duration_sec"] += event["duration_sec"]
        else:
            filtered.append(event)
    return filtered


def summarize_durations(events):
    summary = {}
    for event in events:
        activity = event["activity"]
        if activity not in summary:
            summary[activity] = 0
        summary[activity] += event["duration_sec"]
    return summary


def build_person_summary(track_id, labels, fps, min_duration_frames=3):
    events = extract_events(labels, fps)
    filtered_events = filter_short_events(events, min_duration_frames)
    total_durations = summarize_durations(filtered_events)
    fell = any(event["activity"] == "falling" for event in filtered_events)
    return {
        "track_id": track_id,
        "events": filtered_events,
        "total_durations": total_durations,
        "fell": fell
    }


def predict_dominant_activity(labels, fps, min_duration_frames=3):
    summary = build_person_summary(track_id=0, labels=labels, fps=fps,
                                     min_duration_frames=min_duration_frames)
    if summary["fell"]:
        return "falling"
    total_durations = summary["total_durations"]
    if not total_durations:
        return "unknown"
    return max(total_durations, key=total_durations.get)