"""
solution.py
Sentio Mind · Project 3 · Classroom Engagement Heatmap & Group Analysis

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
# OPTIONAL IMPORTS — graceful fallback if library not installed
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
    _mp_face_mesh   = mp.solutions.face_mesh
    _mp_face_detect = mp.solutions.face_detection
except ImportError:
    _MP_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATH      = Path("video_sample_1.mov")
REPORT_HTML_OUT = Path("engagement_report.html")
OUTPUT_JSON     = Path("engagement_output.json")

WINDOW_SEC     = 6       # seconds per analysis window — keep this configurable
SAMPLE_EVERY_N = 5       # analyse every Nth frame inside a window (speed trade-off)
GRID_ROWS      = 4
GRID_COLS      = 6

# Haar cascade path (bundled with OpenCV)
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_HAAR_PATH)

# ---------------------------------------------------------------------------
# GAZE & EYE OPENNESS  (MediaPipe Face Mesh preferred, fallback brightness)
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh landmark indices
_LEFT_IRIS   = [468, 469, 470, 471, 472]   # left iris centre + ring
_RIGHT_IRIS  = [473, 474, 475, 476, 477]
_LEFT_EYE    = [33, 160, 158, 133, 153, 144]   # outer, top, top, inner, bot, bot
_RIGHT_EYE   = [362, 385, 387, 263, 373, 380]

# For eye-openness: vertical pairs
_L_EYE_TOP   = [160, 158]
_L_EYE_BOT   = [144, 153]
_R_EYE_TOP   = [385, 387]
_R_EYE_BOT   = [373, 380]
_L_EYE_CORN  = [33, 133]   # horizontal extent
_R_EYE_CORN  = [362, 263]


def _landmark_to_np(lm_list, indices, w, h):
    """Convert MediaPipe landmark indices → (N, 2) pixel array."""
    return np.array([[lm_list[i].x * w, lm_list[i].y * h] for i in indices],
                    dtype=np.float32)


def estimate_gaze(face_crop: np.ndarray) -> str:
    """
    Return one of: "forward", "down", "left", "right", "unknown"

    Strategy 1 (MediaPipe iris): normalised iris offset from eye centre.
    Strategy 2 (fallback brightness gradient): dark region position in eye strip.
    """
    h, w = face_crop.shape[:2]
    if h < 20 or w < 20:
        return "unknown"

    # ---- Strategy 1: MediaPipe Face Mesh iris ----
    if _MP_AVAILABLE:
        try:
            with _mp_face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,       # enables iris landmarks 468-477
                max_num_faces=1,
                min_detection_confidence=0.4
            ) as fm:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark

                    # Left iris centre
                    l_iris = _landmark_to_np(lm, _LEFT_IRIS[:1], w, h)[0]
                    l_eye  = _landmark_to_np(lm, _LEFT_EYE, w, h)
                    l_eye_cx = l_eye[:, 0].mean()
                    l_eye_cy = l_eye[:, 1].mean()
                    l_eye_w  = max(l_eye[:, 0].max() - l_eye[:, 0].min(), 1)
                    l_eye_h  = max(l_eye[:, 1].max() - l_eye[:, 1].min(), 1)

                    # Right iris centre
                    r_iris = _landmark_to_np(lm, _RIGHT_IRIS[:1], w, h)[0]
                    r_eye  = _landmark_to_np(lm, _RIGHT_EYE, w, h)
                    r_eye_cx = r_eye[:, 0].mean()
                    r_eye_cy = r_eye[:, 1].mean()
                    r_eye_w  = max(r_eye[:, 0].max() - r_eye[:, 0].min(), 1)
                    r_eye_h  = max(r_eye[:, 1].max() - r_eye[:, 1].min(), 1)

                    # Normalised offsets (avg both eyes)
                    dx = ((l_iris[0] - l_eye_cx) / l_eye_w +
                          (r_iris[0] - r_eye_cx) / r_eye_w) / 2.0
                    dy = ((l_iris[1] - l_eye_cy) / l_eye_h +
                          (r_iris[1] - r_eye_cy) / r_eye_h) / 2.0

                    THRESH_H = 0.10
                    THRESH_V = 0.08
                    if abs(dx) < THRESH_H and abs(dy) < THRESH_V:
                        return "forward"
                    if dy > THRESH_V:
                        return "down"
                    if dy < -THRESH_V:
                        return "forward"   # looking slightly up → still forward
                    if dx < -THRESH_H:
                        return "left"
                    if dx > THRESH_H:
                        return "right"
                    return "forward"
        except Exception:
            pass

    # ---- Strategy 2: brightness-gradient fallback ----
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    # Upper-middle strip = eye region (25–50% height)
    strip = gray[int(h * 0.25): int(h * 0.50), :]
    if strip.size == 0:
        return "unknown"
    # Find darkest column region
    col_mean = strip.mean(axis=0)
    dark_col = int(np.argmin(col_mean))
    norm_x   = dark_col / max(w - 1, 1)   # 0 = left, 1 = right

    # Vertical: compare upper vs lower half of strip
    row_mean   = strip.mean(axis=1)
    upper_dark = row_mean[:len(row_mean) // 2].mean()
    lower_dark = row_mean[len(row_mean) // 2:].mean()

    if lower_dark < upper_dark - 5:
        return "down"
    if norm_x < 0.38:
        return "left"
    if norm_x > 0.62:
        return "right"
    return "forward"


def estimate_eye_openness(face_crop: np.ndarray) -> float:
    """
    Eye height / eye width ratio from face mesh, scaled 0–100.
    Returns 50.0 if landmarks unavailable.
    """
    h, w = face_crop.shape[:2]
    if h < 20 or w < 20:
        return 50.0

    if _MP_AVAILABLE:
        try:
            with _mp_face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.4
            ) as fm:
                rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark

                    def _ear(top_ids, bot_ids, corn_ids):
                        top = _landmark_to_np(lm, top_ids, w, h)[:, 1].mean()
                        bot = _landmark_to_np(lm, bot_ids, w, h)[:, 1].mean()
                        corn = _landmark_to_np(lm, corn_ids, w, h)
                        eye_w = max(abs(corn[0, 0] - corn[1, 0]), 1)
                        return abs(bot - top) / eye_w   # eye-aspect-ratio

                    l_ear = _ear(_L_EYE_TOP, _L_EYE_BOT, _L_EYE_CORN)
                    r_ear = _ear(_R_EYE_TOP, _R_EYE_BOT, _R_EYE_CORN)
                    ear   = (l_ear + r_ear) / 2.0

                    # Typical open EAR ≈ 0.25–0.35, closed ≈ 0.05–0.15
                    score = np.clip((ear - 0.05) / (0.35 - 0.05), 0.0, 1.0) * 100
                    return float(score)
        except Exception:
            pass

    return 50.0


# ---------------------------------------------------------------------------
# PER-FACE ENGAGEMENT SCORE
# ---------------------------------------------------------------------------

def score_face(face_crop: np.ndarray) -> dict:
    """
    Compute engagement for one face.
    Returns: { "score": float, "gaze": str, "eye_openness": float }

    Formula:
        face_engagement = (forward_gaze × 40) + (eye_openness × 0.25) + (head_pose × 0.35)
        forward_gaze = 1  if gaze == "forward" else 0.25
        head_pose    = 1.0 if gaze == "forward" else 0.3   (gaze-based proxy)
    """
    gaze         = estimate_gaze(face_crop)
    eye_openness = estimate_eye_openness(face_crop)

    forward_gaze = 1.0 if gaze == "forward" else 0.25
    head_pose    = 1.0 if gaze == "forward" else 0.3

    raw_score = (forward_gaze * 40) + (eye_openness * 0.25) + (head_pose * 0.35)
    score     = float(np.clip(raw_score, 0, 100))

    return {"score": score, "gaze": gaze, "eye_openness": eye_openness}


# ---------------------------------------------------------------------------
# SPATIAL ZONE
# ---------------------------------------------------------------------------

def get_zone(bbox: tuple, frame_w: int, frame_h: int) -> str:
    """
    bbox = (x, y, w, h). Use face centre.
    Return zone like "R2C3". Clamp to valid grid.
    """
    x, y, bw, bh = bbox
    cx = x + bw / 2
    cy = y + bh / 2

    col = int((cx / frame_w) * GRID_COLS)
    row = int((cy / frame_h) * GRID_ROWS)

    col = min(max(col, 0), GRID_COLS - 1)
    row = min(max(row, 0), GRID_ROWS - 1)

    return f"R{row + 1}C{col + 1}"


# ---------------------------------------------------------------------------
# FACE DETECTION IN ONE FRAME
# ---------------------------------------------------------------------------

def detect_faces(frame: np.ndarray) -> list:
    """
    Detect all faces. Returns: [{"bbox": (x,y,w,h), "face_crop": ndarray}, ...]
    Applies CLAHE first. Uses MediaPipe (preferred) or Haar cascade (fallback).
    """
    # CLAHE pre-processing
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    detections = []
    fh, fw = frame.shape[:2]

    # ---- MediaPipe Face Detection ----
    if _MP_AVAILABLE:
        try:
            with _mp_face_detect.FaceDetection(
                model_selection=1,           # 1 = full-range (≥2 m distance)
                min_detection_confidence=0.45
            ) as fd:
                rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                res = fd.process(rgb)
                if res.detections:
                    for det in res.detections:
                        bb = det.location_data.relative_bounding_box
                        x  = max(int(bb.xmin * fw), 0)
                        y  = max(int(bb.ymin * fh), 0)
                        bw = min(int(bb.width  * fw), fw - x)
                        bh = min(int(bb.height * fh), fh - y)
                        if bw > 10 and bh > 10:
                            crop = frame[y: y + bh, x: x + bw]
                            detections.append({"bbox": (x, y, bw, bh),
                                               "face_crop": crop})
            return detections
        except Exception:
            pass

    # ---- Haar cascade fallback ----
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )
    for (x, y, bw, bh) in (faces if len(faces) else []):
        crop = frame[y: y + bh, x: x + bw]
        detections.append({"bbox": (x, y, bw, bh), "face_crop": crop})

    return detections


# ---------------------------------------------------------------------------
# PROCESS ONE TIME WINDOW
# ---------------------------------------------------------------------------

def process_window(frames_in_window: list, frame_w: int, frame_h: int) -> dict:
    """
    Aggregate face engagement across sampled frames in one window.
    """
    all_scores      = []
    gaze_counts     = {"forward": 0, "down": 0, "left": 0, "right": 0, "unknown": 0}
    zone_scores_map = defaultdict(list)
    face_crops_out  = []

    sampled = frames_in_window[::SAMPLE_EVERY_N] if len(frames_in_window) > SAMPLE_EVERY_N else frames_in_window

    for (_, _, frame) in sampled:
        dets = detect_faces(frame)
        for det in dets:
            crop   = det["face_crop"]
            result = score_face(crop)
            zone   = get_zone(det["bbox"], frame_w, frame_h)

            all_scores.append(result["score"])
            gaze_counts[result["gaze"]] += 1
            zone_scores_map[zone].append(result["score"])

            # Keep small crops for collage (max 10 per window)
            if len(face_crops_out) < 10:
                face_crops_out.append(crop)

    engagement_score = float(np.mean(all_scores)) if all_scores else 0.0
    spatial_zones    = {z: float(np.mean(v)) for z, v in zone_scores_map.items()}

    return {
        "engagement_score":  round(engagement_score, 2),
        "persons_count":     len(all_scores),
        "gaze_distribution": dict(gaze_counts),
        "spatial_zones":     spatial_zones,
        "face_crops":        face_crops_out,
    }


# ---------------------------------------------------------------------------
# COLLAGE HELPER
# ---------------------------------------------------------------------------

def make_collage_b64(face_crops: list, cell: int = 60) -> str:
    """
    Stack face crops horizontally into a small JPEG collage.
    Returns base64-encoded JPEG string, or "" if no crops.
    """
    if not face_crops:
        return ""

    cells = []
    for crop in face_crops[:8]:    # cap at 8 thumbnails
        if crop is None or crop.size == 0:
            continue
        resized = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
        cells.append(resized)

    if not cells:
        return ""

    collage = np.hstack(cells)
    ok, buf = cv2.imencode(".jpg", collage, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# HTML REPORT  (fully offline — Chart.js inlined via CDN-free UMD bundle)
# ---------------------------------------------------------------------------

# Minimal Chart.js 4 UMD (minified) — embedded so report works offline.
# We fetch it once from unpkg at build time and inline it, OR use the tiny
# canvas-based fallback below if network is not available.
def _try_get_chartjs() -> str:
    """Return Chart.js UMD source as a string, or '' to fall back to plain bars."""
    try:
        import urllib.request
        url = "https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode("utf-8")
    except Exception:
        return ""


def _color_for_score(score: float) -> str:
    """Return hex colour: green ≥70, amber 50–70, red <50."""
    if score >= 70:
        return "#22c55e"
    if score >= 50:
        return "#f59e0b"
    return "#ef4444"


def _zone_color(score: float) -> str:
    """Linear interpolation: red(0) → white(50) → green(100)."""
    s = max(0.0, min(100.0, float(score)))
    if s <= 50:
        t  = s / 50.0
        r  = int(239 + (255 - 239) * t)
        g  = int(68  + (255 - 68)  * t)
        b  = int(68  + (255 - 68)  * t)
    else:
        t  = (s - 50) / 50.0
        r  = int(255 + (34  - 255) * t)
        g  = int(255 + (197 - 255) * t)
        b  = int(255 + (94  - 255) * t)
    return f"rgb({r},{g},{b})"


def generate_engagement_report(windows: list, stats: dict, output_path: Path):
    """
    Write a fully self-contained engagement_report.html with:
      1. Session summary cards
      2. Engagement timeline (Chart.js or plain bars)
      3. 4×6 spatial heatmap
      4. Worst-3-window evidence panels
    """
    chartjs_src = _try_get_chartjs()
    use_chartjs = bool(chartjs_src)

    overall  = stats.get("overall_engagement_score", 0)
    peak_w   = stats.get("peak_window",   {})
    trough_w = stats.get("trough_window", {})
    total_p  = stats.get("total_persons_detected", 0)
    dur      = stats.get("session_duration_sec", 0)
    worst3   = stats.get("worst_3_windows", [])
    heatmap  = stats.get("session_spatial_heatmap", {})

    # Timeline data
    labels   = [f"{w.get('start_sec', 0) / 60:.1f}m" for w in windows]
    scores   = [w.get("engagement_score", 0) for w in windows]
    colors   = [_color_for_score(s) for s in scores]

    # Worst-3 evidence HTML
    worst_html = ""
    for i, w in enumerate(worst3, 1):
        ts       = w.get("start_sec", 0)
        score    = w.get("score", 0)
        b64      = w.get("thumbnail_collage_b64", "")
        img_tag  = (f'<img src="data:image/jpeg;base64,{b64}" '
                    f'style="max-width:100%;border-radius:6px;" alt="faces">')
        if not b64:
            img_tag = '<div style="color:#999;font-size:13px;">No face thumbnails captured</div>'

        worst_html += f"""
        <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;
                    padding:16px;flex:1;min-width:220px;">
          <div style="font-size:13px;color:#6b7280;margin-bottom:4px;">
            Worst Window #{i} &nbsp;|&nbsp; ⏱ {ts:.0f}s
          </div>
          <div style="font-size:28px;font-weight:700;color:{_color_for_score(score)};">
            {score:.0f}<span style="font-size:16px;font-weight:400;">%</span>
          </div>
          <div style="margin-top:10px;">{img_tag}</div>
        </div>"""

    # Heatmap HTML
    heatmap_rows = ""
    for r in range(1, GRID_ROWS + 1):
        heatmap_rows += "<tr>"
        for c in range(1, GRID_COLS + 1):
            key   = f"R{r}C{c}"
            s     = heatmap.get(key, 0)
            bg    = _zone_color(s)
            text_c = "#111" if 30 < s < 75 else ("#fff" if s <= 30 else "#111")
            heatmap_rows += (
                f'<td style="background:{bg};color:{text_c};width:60px;height:46px;'
                f'text-align:center;font-size:12px;font-weight:600;'
                f'border:1px solid #e5e7eb;">'
                f'{key}<br><span style="font-size:10px;">{s}%</span></td>'
            )
        heatmap_rows += "</tr>"

    # Chart section — Chart.js or plain horizontal bars
    if use_chartjs:
        chart_section = f"""
        <script>{chartjs_src}</script>
        <canvas id="engChart" height="90"></canvas>
        <script>
        (function(){{
          const labels = {json.dumps(labels)};
          const scores = {json.dumps(scores)};
          const colors = {json.dumps(colors)};
          new Chart(document.getElementById('engChart'), {{
            type: 'line',
            data: {{
              labels: labels,
              datasets: [{{
                label: 'Engagement %',
                data: scores,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99,102,241,0.08)',
                borderWidth: 2,
                pointBackgroundColor: colors,
                pointRadius: 4,
                tension: 0.35,
                fill: true,
              }}]
            }},
            options: {{
              scales: {{
                y: {{ min: 0, max: 100,
                      title: {{ display: true, text: 'Engagement (%)' }} }},
                x: {{ title: {{ display: true, text: 'Time (minutes)' }} }}
              }},
              plugins: {{
                legend: {{ display: false }},
                tooltip: {{ callbacks: {{
                  label: ctx => ` ${{ctx.parsed.y.toFixed(1)}}%`
                }} }}
              }},
              animation: false,
              responsive: true
            }}
          }});
        }})();
        </script>"""
    else:
        # Plain HTML bars fallback
        bar_items = ""
        for lbl, sc, clr in zip(labels, scores, colors):
            bar_items += (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                f'<span style="width:40px;font-size:11px;color:#6b7280;">{lbl}</span>'
                f'<div style="flex:1;background:#f3f4f6;border-radius:3px;height:14px;">'
                f'<div style="width:{sc}%;background:{clr};height:14px;border-radius:3px;"></div></div>'
                f'<span style="width:34px;font-size:11px;text-align:right;">{sc:.0f}%</span>'
                f'</div>'
            )
        chart_section = f'<div style="padding:8px 0;">{bar_items}</div>'

    # Gaze distribution summary (aggregate)
    total_gaze = defaultdict(int)
    for w in windows:
        for g, cnt in w.get("gaze_distribution", {}).items():
            total_gaze[g] += cnt
    total_g_sum = sum(total_gaze.values()) or 1
    gaze_bars = ""
    gaze_icons = {"forward": "👁️", "down": "⬇️", "left": "⬅️", "right": "➡️", "unknown": "❓"}
    for g in ["forward", "down", "left", "right", "unknown"]:
        pct = total_gaze[g] / total_g_sum * 100
        gaze_bars += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="width:70px;font-size:12px;">{gaze_icons.get(g,"")} {g.title()}</span>'
            f'<div style="flex:1;background:#f3f4f6;border-radius:3px;height:14px;">'
            f'<div style="width:{pct:.1f}%;background:#6366f1;height:14px;border-radius:3px;"></div></div>'
            f'<span style="width:38px;font-size:12px;text-align:right;">{pct:.1f}%</span>'
            f'</div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Classroom Engagement Report · Sentio Mind</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: #f8fafc; color: #1e293b; min-height: 100vh; }}
  .topbar {{ background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
             color: #fff; padding: 20px 32px; }}
  .topbar h1 {{ font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }}
  .topbar p  {{ font-size: 13px; opacity: 0.82; margin-top: 2px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 28px 24px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
           padding: 20px 24px; margin-bottom: 24px; }}
  .card h2 {{ font-size: 15px; font-weight: 700; color: #374151; margin-bottom: 14px;
              text-transform: uppercase; letter-spacing: 0.04em; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                   gap: 16px; margin-bottom: 24px; }}
  .summary-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
                   padding: 18px 20px; }}
  .summary-card .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase;
                          letter-spacing: 0.06em; margin-bottom: 6px; }}
  .summary-card .value {{ font-size: 30px; font-weight: 800; line-height: 1; }}
  .summary-card .sub   {{ font-size: 12px; color: #9ca3af; margin-top: 4px; }}
  .worst-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  table.heatmap {{ border-collapse: collapse; width: 100%; }}
  .legend {{ display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size:12px; }}
  .legend-bar {{ flex: 1; height: 14px; border-radius: 4px;
                  background: linear-gradient(to right, #ef4444, #ffffff 50%, #22c55e); }}
</style>
</head>
<body>

<div class="topbar">
  <h1>🎓 Classroom Engagement Report</h1>
  <p>Sentio Mind · {stats.get('video','video_sample_1.mov')} · {stats.get('date', str(date.today()))}
     · {dur / 60:.1f} min session</p>
</div>

<div class="container">

  <!-- SUMMARY CARDS -->
  <div class="summary-grid">
    <div class="summary-card">
      <div class="label">Overall Score</div>
      <div class="value" style="color:{_color_for_score(overall)};">{overall}%</div>
      <div class="sub">session average</div>
    </div>
    <div class="summary-card">
      <div class="label">Peak Window</div>
      <div class="value" style="color:#22c55e;">{peak_w.get('engagement_score', peak_w.get('score', 0)):.0f}%</div>
      <div class="sub">Window #{peak_w.get('window_id','?')} @ {peak_w.get('start_sec',0):.0f}s</div>
    </div>
    <div class="summary-card">
      <div class="label">Trough Window</div>
      <div class="value" style="color:#ef4444;">{trough_w.get('engagement_score', trough_w.get('score', 0)):.0f}%</div>
      <div class="sub">Window #{trough_w.get('window_id','?')} @ {trough_w.get('start_sec',0):.0f}s</div>
    </div>
    <div class="summary-card">
      <div class="label">Total Detections</div>
      <div class="value" style="color:#6366f1;">{total_p}</div>
      <div class="sub">face·frame observations</div>
    </div>
    <div class="summary-card">
      <div class="label">Windows Analysed</div>
      <div class="value" style="color:#0ea5e9;">{len(windows)}</div>
      <div class="sub">× {WINDOW_SEC}s each</div>
    </div>
  </div>

  <!-- TIMELINE CHART -->
  <div class="card">
    <h2>📈 Engagement Timeline</h2>
    {chart_section}
    <div style="display:flex;gap:16px;margin-top:10px;font-size:12px;">
      <span style="display:flex;align-items:center;gap:4px;">
        <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#22c55e;"></span> ≥70 % engaged
      </span>
      <span style="display:flex;align-items:center;gap:4px;">
        <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#f59e0b;"></span> 50–70% moderate
      </span>
      <span style="display:flex;align-items:center;gap:4px;">
        <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#ef4444;"></span> &lt;50% disengaged
      </span>
    </div>
  </div>

  <!-- SPATIAL HEATMAP -->
  <div class="card">
    <h2>🗺️ Spatial Engagement Heatmap (Session Average)</h2>
    <p style="font-size:12px;color:#6b7280;margin-bottom:12px;">
      Rows = front→back of classroom &nbsp;|&nbsp; Columns = left→right seating zones
    </p>
    <div style="overflow-x:auto;">
      <table class="heatmap">
        <tbody>{heatmap_rows}</tbody>
      </table>
    </div>
    <div class="legend">
      <span>0%</span>
      <div class="legend-bar"></div>
      <span>100%</span>
    </div>
  </div>

  <!-- GAZE DISTRIBUTION -->
  <div class="card">
    <h2>👀 Session Gaze Distribution</h2>
    {gaze_bars}
  </div>

  <!-- WORST 3 WINDOWS -->
  <div class="card">
    <h2>🔴 Worst 3 Engagement Windows (Evidence)</h2>
    <p style="font-size:12px;color:#6b7280;margin-bottom:14px;">
      Face thumbnail collages from the lowest-engagement windows for counsellor review.
    </p>
    <div class="worst-row">
      {worst_html if worst_html else '<p style="color:#9ca3af;font-size:13px;">No window data available.</p>'}
    </div>
  </div>

</div><!-- /container -->

<footer style="text-align:center;padding:20px;font-size:12px;color:#9ca3af;">
  Generated by Sentio Mind · Project 3 · Classroom Engagement Heatmap
</footer>

</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  HTML report written → {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video file not found: {VIDEO_PATH}")
        print("  Place video_sample_1.mov in the same directory as solution.py and re-run.")
        raise SystemExit(1)

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
    print(f"Frame size: {fw}×{fh}  |  FPS: {fps:.1f}  |  MediaPipe: {_MP_AVAILABLE}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps

        # Collect every SAMPLE_EVERY_N-th frame into the buffer
        if frame_idx % SAMPLE_EVERY_N == 0:
            window_buf.append((frame_idx, ts, frame.copy()))

        # End of window OR last frame
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
            wid = result["window_id"]
            if wid % 5 == 0 or wid == 1:
                print(f"  Window {wid:3d}  |  {ts:6.1f}s  |  "
                      f"score={result['engagement_score']:.0f}%  |  "
                      f"persons={result['persons_count']}")

        frame_idx += 1
    cap.release()

    # ---- Session spatial heatmap (aggregate all windows) ----
    zone_scores = defaultdict(list)
    for w in windows:
        for zid, score in w["spatial_zones"].items():
            zone_scores[zid].append(score)
    heatmap = {
        f"R{r+1}C{c+1}": int(np.mean(zone_scores[f"R{r+1}C{c+1}"]))
        if zone_scores[f"R{r+1}C{c+1}"] else 0
        for r in range(GRID_ROWS) for c in range(GRID_COLS)
    }

    # ---- Derived stats ----
    scores   = [w["engagement_score"] for w in windows]
    peak     = max(windows, key=lambda w: w["engagement_score"])
    trough   = min(windows, key=lambda w: w["engagement_score"])
    worst3   = sorted(windows, key=lambda w: w["engagement_score"])[:3]

    for w in worst3:
        w["thumbnail_collage_b64"] = make_collage_b64(w.get("face_crops", []))

    def clean(w: dict) -> dict:
        """Strip internal-only keys before JSON serialisation."""
        return {k: v for k, v in w.items() if k != "face_crops"}

    def dominant_gaze(w: dict) -> str:
        gd = w.get("gaze_distribution", {})
        return max(gd, key=gd.get) if gd else "unknown"

    stats = {
        "source":                   "p3_classroom_engagement",
        "video":                    str(VIDEO_PATH),
        "date":                     str(date.today()),
        "session_duration_sec":     round(dur, 2),
        "window_size_sec":          WINDOW_SEC,
        "total_persons_detected":   sum(w["persons_count"] for w in windows),
        "overall_engagement_score": int(np.mean(scores)) if scores else 0,
        "peak_window": {
            "window_id":      peak["window_id"],
            "start_sec":      peak["start_sec"],
            "end_sec":        peak["end_sec"],
            "score":          peak["engagement_score"],
            "dominant_gaze":  dominant_gaze(peak),
        },
        "trough_window": {
            "window_id":      trough["window_id"],
            "start_sec":      trough["start_sec"],
            "end_sec":        trough["end_sec"],
            "score":          trough["engagement_score"],
            "dominant_gaze":  dominant_gaze(trough),
        },
        "worst_3_windows": [
            {
                "window_id":              w["window_id"],
                "start_sec":              w["start_sec"],
                "end_sec":                w["end_sec"],
                "score":                  w["engagement_score"],
                "thumbnail_collage_b64":  w.get("thumbnail_collage_b64", ""),
            }
            for w in worst3
        ],
        "time_windows":            [clean(w) for w in windows],
        "session_spatial_heatmap": heatmap,
    }

    # ---- Write JSON ----
    with open(OUTPUT_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  JSON written → {OUTPUT_JSON}")

    # ---- Generate HTML report ----
    generate_engagement_report(windows, stats, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Overall engagement:  {stats['overall_engagement_score']}%")
    print(f"  Peak:   window {peak['window_id']:3d}  @{peak['start_sec']:6.0f}s"
          f"  →  {peak['engagement_score']:.1f}%")
    print(f"  Trough: window {trough['window_id']:3d}  @{trough['start_sec']:6.0f}s"
          f"  →  {trough['engagement_score']:.1f}%")
    print(f"  Windows: {len(windows)}  |  Persons: {stats['total_persons_detected']}")
    print(f"  Report → {REPORT_HTML_OUT}")
    print(f"  JSON   → {OUTPUT_JSON}")
    print("=" * 55)
