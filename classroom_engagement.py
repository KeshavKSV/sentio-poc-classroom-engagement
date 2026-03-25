"""
classroom_engagement.py
Sentio Mind · Project 3 · Classroom Engagement Heatmap & Group Analysis

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
"""

import cv2
import json
import base64
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path("video_sample_1.mov")
REPORT_HTML_OUT = Path("engagement_report.html")
OUTPUT_JSON     = Path("engagement_output.json")

WINDOW_SEC      = 6       # seconds per analysis window — keep this configurable
SAMPLE_EVERY_N  = 5       # analyse every Nth frame inside a window (speed trade-off)
GRID_ROWS       = 4
GRID_COLS       = 6


# ---------------------------------------------------------------------------
# GAZE & EYE OPENNESS
# ---------------------------------------------------------------------------

def estimate_gaze(face_crop: np.ndarray) -> str:
    """
    Return one of: "forward", "down", "left", "right", "unknown"

    Preferred: MediaPipe Face Mesh iris landmarks — compute offset of iris
    centre from eye centre, normalise to eye width. Threshold at 0.1.

    Fallback: brightness gradient across the eye region — if the dark region
    (iris) is left of centre, gaze is "left" etc.

    TODO: implement
    """
    # TODO
    return "forward"


def estimate_eye_openness(face_crop: np.ndarray) -> float:
    """
    Eye height / eye width ratio from face landmarks, scaled 0–100.
    Return 50.0 if no landmark found.
    TODO: implement
    """
    # TODO
    return 50.0


# ---------------------------------------------------------------------------
# PER-FACE ENGAGEMENT SCORE
# ---------------------------------------------------------------------------

def score_face(face_crop: np.ndarray) -> dict:
    """
    Compute engagement for one face.
    Return: { "score": float, "gaze": str, "eye_openness": float }

    Formula from README:
      face_engagement = (forward_gaze × 40) + (eye_openness × 0.25) + (head_pose × 0.35)
      forward_gaze = 1 if gaze == "forward" else 0.25
      head_pose    = 1.0 if gaze == "forward" else 0.3  (simple proxy)

    TODO: implement
    """
    # TODO
    return {"score": 50.0, "gaze": "unknown", "eye_openness": 50.0}


# ---------------------------------------------------------------------------
# SPATIAL ZONE
# ---------------------------------------------------------------------------

def get_zone(bbox: tuple, frame_w: int, frame_h: int) -> str:
    """
    bbox = (x, y, w, h) in pixels. Use face centre.
    Return zone ID string like "R2C3".
    Clamp column 0–5, row 0–3.
    TODO: implement the formula from README
    """
    # TODO
    return "R1C1"


# ---------------------------------------------------------------------------
# FACE DETECTION IN ONE FRAME
# ---------------------------------------------------------------------------

def detect_faces(frame: np.ndarray) -> list:
    """
    Detect all faces. Return: [{"bbox": (x,y,w,h), "face_crop": ndarray}, ...]
    Apply CLAHE on the frame first.
    Use Haar cascade or MediaPipe Face Detection — whichever works better on your data.
    TODO: implement
    """
    detections = []
    # TODO
    return detections


# ---------------------------------------------------------------------------
# PROCESS ONE TIME WINDOW
# ---------------------------------------------------------------------------

def process_window(frames_in_window: list, frame_w: int, frame_h: int) -> dict:
    """
    frames_in_window: [(frame_idx, timestamp_sec, ndarray), ...]

    For each sampled frame:
      - detect faces
      - score each face
      - assign each face to a spatial zone

    Aggregate:
      - group engagement_score = mean of all face scores
      - gaze_distribution count
      - spatial_zones = mean score per zone

    Return a dict that becomes one entry in time_windows.
    Also attach face_crops list (a few small crops for evidence thumbnails — not in JSON).

    TODO: implement
    """
    # TODO
    return {
        "engagement_score":  0,
        "persons_count":     0,
        "gaze_distribution": {"forward": 0, "down": 0, "left": 0, "right": 0, "unknown": 0},
        "spatial_zones":     {},
        "face_crops":        [],    # internal only, stripped before writing JSON
    }


# ---------------------------------------------------------------------------
# COLLAGE HELPER
# ---------------------------------------------------------------------------

def make_collage_b64(face_crops: list, cell: int = 60) -> str:
    """
    Stack face crops horizontally into a small collage, return as base64 JPEG.
    Resize each crop to cell × cell before stacking.
    Return empty string if no crops.
    TODO: implement
    """
    if not face_crops:
        return ""
    # TODO
    return ""


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

def generate_engagement_report(windows: list, stats: dict, output_path: Path):
    """
    Write engagement_report.html — self-contained, no CDN.

    Must include:
      1. Session summary numbers
      2. Engagement timeline chart  (use Chart.js bundled inline OR plain HTML bars)
      3. 4×6 heatmap: HTML table, cell background-color based on zone score
         colour scale: red (#ef4444) at 0, white at 50, green (#22c55e) at 100
      4. 3 worst windows: timestamp + score + face collage image

    For Chart.js offline: download chart.umd.min.js and paste between <script> tags.
    TODO: implement
    """
    # TODO
    pass


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cap   = cv2.VideoCapture(str(VIDEO_PATH))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur   = total / fps

    frames_per_window = max(1, int(WINDOW_SEC * fps))
    windows     = []
    window_buf  = []
    w_start_sec = 0.0
    frame_idx   = 0

    print(f"Processing {VIDEO_PATH}  |  {dur:.1f}s  |  {total} frames  |  window={WINDOW_SEC}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps

        if frame_idx % SAMPLE_EVERY_N == 0:
            window_buf.append((frame_idx, ts, frame.copy()))

        if (frame_idx + 1) % frames_per_window == 0 or frame_idx == total - 1:
            result = process_window(window_buf, fw, fh)
            result.update({
                "window_id": len(windows) + 1,
                "start_sec": round(w_start_sec, 2),
                "end_sec":   round(ts, 2),
            })
            windows.append(result)
            window_buf  = []
            w_start_sec = ts + 1 / fps

        frame_idx += 1
    cap.release()

    # Session heatmap
    zone_scores = defaultdict(list)
    for w in windows:
        for zid, score in w["spatial_zones"].items():
            zone_scores[zid].append(score)
    heatmap = {
        f"R{r+1}C{c+1}": int(np.mean(zone_scores[f"R{r+1}C{c+1}"])) if zone_scores[f"R{r+1}C{c+1}"] else 0
        for r in range(GRID_ROWS) for c in range(GRID_COLS)
    }

    scores   = [w["engagement_score"] for w in windows]
    peak     = max(windows, key=lambda w: w["engagement_score"])
    trough   = min(windows, key=lambda w: w["engagement_score"])
    worst3   = sorted(windows, key=lambda w: w["engagement_score"])[:3]

    for w in worst3:
        w["thumbnail_collage_b64"] = make_collage_b64(w.get("face_crops", []))

    def clean(w):
        return {k: v for k, v in w.items() if k != "face_crops"}

    stats = {
        "source":                   "p3_classroom_engagement",
        "video":                    str(VIDEO_PATH),
        "date":                     str(date.today()),
        "session_duration_sec":     round(dur, 2),
        "window_size_sec":          WINDOW_SEC,
        "total_persons_detected":   sum(w["persons_count"] for w in windows),
        "overall_engagement_score": int(np.mean(scores)) if scores else 0,
        "peak_window":    clean(peak),
        "trough_window":  clean(trough),
        "worst_3_windows": [
            {"window_id": w["window_id"], "start_sec": w["start_sec"],
             "end_sec": w["end_sec"], "score": w["engagement_score"],
             "thumbnail_collage_b64": w.get("thumbnail_collage_b64", "")}
            for w in worst3
        ],
        "time_windows":             [clean(w) for w in windows],
        "session_spatial_heatmap":  heatmap,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    generate_engagement_report(windows, stats, REPORT_HTML_OUT)

    print()
    print("=" * 50)
    print(f"  Overall engagement:  {stats['overall_engagement_score']}%")
    print(f"  Peak:    window {peak['window_id']}  @ {peak['start_sec']:.0f}s  =  {peak['engagement_score']}%")
    print(f"  Trough:  window {trough['window_id']}  @ {trough['start_sec']:.0f}s  =  {trough['engagement_score']}%")
    print(f"  Report → {REPORT_HTML_OUT}")
    print(f"  JSON   → {OUTPUT_JSON}")
    print("=" * 50)
