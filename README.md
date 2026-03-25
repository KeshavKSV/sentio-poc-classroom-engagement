# Classroom Engagement Heatmap & Group Analysis
**Sentio Mind · POC Assignment · Project 3**

GitHub: https://github.com/Sentiodirector/sentio-poc-classroom-engagement.git
Branch: FirstName_LastName_RollNumber

---

## Why This Exists

Sentio Mind tracks individuals. But teachers want to understand the room. Which 6-second window had the highest collective attention? Which seats are chronically disengaged? Is the class losing focus after 20 minutes? None of that is possible right now. Your job is to build it.

---

## What You Receive

```
p3_classroom_engagement/
├── video_sample_1.mov            ← download from dataset link in your assignment
├── classroom_engagement.py       ← your template — copy to solution.py
├── classroom_engagement.json     ← schema for engagement_output.json
└── README.md
```

---

## What You Must Build

Run `python solution.py` → it must produce:

1. `engagement_report.html` — dashboard with timeline chart, heatmap, and evidence thumbnails
2. `engagement_output.json` — follows `classroom_engagement.json` schema exactly

### How Scoring Works

For each face detected in a window:
```
face_engagement = (forward_gaze × 40) + (eye_openness × 0.25) + (head_pose × 0.35)
```
- `forward_gaze` — 1 if gaze is "forward", else 0.25
- `eye_openness` — 0 to 100 from face mesh, used directly
- `head_pose` — 1.0 if head roughly faces camera, else 0.3

Group score for window = mean of all face scores in that window. If no faces: score = 0.

### Spatial Zone Assignment

```python
zone_col = int((face_center_x / frame_width)  * 6)   # 0–5
zone_row = int((face_center_y / frame_height) * 4)   # 0–3
zone_id  = f"R{zone_row + 1}C{zone_col + 1}"
```

Clamp column to max 5, row to max 3.

### What the Report Must Show

1. Session summary: overall score, peak window, trough window, total persons
2. Line chart: X = time in minutes, Y = engagement 0–100 (green ≥70, amber 50–70, red <50)
3. Heatmap: 4-row × 6-column grid, each cell coloured by zone engagement score
4. Evidence: for the 3 worst windows — timestamp, score, face thumbnail collage

---

## Hard Rules

- Do not rename functions in `classroom_engagement.py`
- Do not change key names in `classroom_engagement.json`
- Window size is configurable — default 6 seconds
- `engagement_report.html` must work offline, no CDN
- Python 3.9+, no Jupyter notebooks

## Libraries

```
opencv-python==4.9.0   mediapipe==0.10.14   numpy==1.26.4   Pillow==10.3.0
```

---

## Submit

| # | File | What |
|---|------|------|
| 1 | `solution.py` | Working script |
| 2 | `engagement_report.html` | Full dashboard |
| 3 | `engagement_output.json` | Output matching schema |
| 4 | `demo.mp4` | Screen recording under 2 min |

Push to your branch only. Do not touch main.

---

## Bonus

Identify which zone likely contains the teacher (usually has the most detections and highest motion variance) and mark it differently in the heatmap.

*Sentio Mind · 2026*
