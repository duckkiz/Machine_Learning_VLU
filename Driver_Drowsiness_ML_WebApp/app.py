"""
Driver Drowsiness Detection - Flask Web Application
====================================================
Run:  python app.py
Then: http://localhost:5000
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

WEIGHTS_PATH = MODELS_DIR / "best.pt"
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
VISUAL_ALERT_LABELS = {"drowsy", "yawning", "looking_away", "distracted"}
SOUND_ALERT_LABELS  = {"drowsy", "yawning", "distracted"}
REAL_LABELS         = {"awake", "drowsy", "yawning", "looking_away"}

LABEL_COLORS_HEX = {
    "awake":        "#00e676",
    "drowsy":       "#ff1744",
    "yawning":      "#ff9100",
    "looking_away": "#2979ff",
    "distracted":   "#d500f9",
    "too_far":      "#ffd740",
    "uncertain":    "#90a4ae",
    "no_face":      "#607d8b",
}

# ──────────────────────────────────────────────
# Global detection state (thread-safe via lock)
# ──────────────────────────────────────────────
state_lock = threading.Lock()
detection_state: dict = {
    "label":       "no_face",
    "confidence":  0.0,
    "alert":       False,
    "note":        "",
    "fps":         0.0,
    "color":       LABEL_COLORS_HEX["no_face"],
    "running":     False,
    "alert_history": [],   # list of {"time": str, "label": str}
}

camera_thread: threading.Thread | None = None
stop_event = threading.Event()
output_frame_lock = threading.Lock()
output_frame: np.ndarray | None = None


# ──────────────────────────────────────────────
# Detection helpers (ported from detect_webcam.py)
# ──────────────────────────────────────────────
def classify_roi(model: YOLO, roi_bgr, imgsz: int = 224):
    results = model.predict(source=roi_bgr, imgsz=imgsz, verbose=False)
    if not results:
        return "uncertain", 0.0
    res = results[0]
    probs = getattr(res, "probs", None)
    if probs is None:
        return "uncertain", 0.0
    top1_idx = int(probs.top1)
    conf     = float(probs.top1conf)
    label    = res.names[top1_idx]
    return label, conf


def stable_label(history: deque, min_votes: int = 6):
    valid = [lbl for lbl, _ in history if lbl in REAL_LABELS]
    if not valid:
        return "uncertain", 0.0
    counts = Counter(valid)
    label, votes = counts.most_common(1)[0]
    if votes < min_votes:
        return "uncertain", 0.0
    confs = [c for lbl, c in history if lbl == label]
    return label, (sum(confs) / len(confs) if confs else 0.0)


def crop_face(frame_bgr, landmarks, margin: float = 0.25):
    h, w = frame_bgr.shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = int(max(0.0, min(xs) - margin) * w)
    y1 = int(max(0.0, min(ys) - margin) * h)
    x2 = int(min(1.0, max(xs) + margin) * w)
    y2 = int(min(1.0, max(ys) + margin) * h)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def color_bgr(label: str):
    return {
        "awake":        (0, 230, 118),
        "drowsy":       (68, 23, 255),
        "yawning":      (0, 145, 255),
        "looking_away": (255, 121, 41),
        "distracted":   (249, 0, 213),
        "too_far":      (64, 215, 255),
        "uncertain":    (164, 164, 144),
        "no_face":      (120, 125, 96),
    }.get(label, (255, 255, 255))


def draw_overlay(frame, label: str, conf: float, note: str = ""):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 110), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    clr = color_bgr(label)
    cv2.putText(frame, "Driver State",  (22, 36),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(frame, label.upper(),   (22, 72),  cv2.FONT_HERSHEY_SIMPLEX, 0.90, clr, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {conf:.2f}", (22, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
    if note:
        cv2.putText(frame, note, (22, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1, cv2.LINE_AA)
    if label in VISUAL_ALERT_LABELS:
        cv2.putText(frame, "⚠ ALERT", (22, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Camera worker thread
# ──────────────────────────────────────────────
def camera_worker(camera_index: int = 0, window: int = 9, min_votes: int = 6,
                  conf_thresh: float = 0.45, min_face_w: int = 140, min_face_h: int = 140,
                  drowsy_hold: int = 10, yawning_hold: int = 12,
                  missing_thresh: int = 10, reacquire_thresh: int = 2):
    global output_frame

    if not WEIGHTS_PATH.exists() or not FACE_MODEL_PATH.exists():
        with state_lock:
            detection_state["note"] = "Model files not found!"
            detection_state["running"] = False
        return

    cls_model = YOLO(str(WEIGHTS_PATH))

    BaseOptions          = python.BaseOptions
    FaceLandmarker       = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode    = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    history             = deque(maxlen=window)
    face_missing_count  = 0
    face_present_count  = 0
    in_missing_alert    = False
    last_confirmed_lbl  = "awake"
    last_confirmed_conf = 0.0
    drowsy_streak       = 0
    yawning_streak      = 0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        with state_lock:
            detection_state["note"] = "Cannot open webcam!"
            detection_state["running"] = False
        return

    fps_counter  = 0
    fps_start    = time.time()
    current_fps  = 0.0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms     = int(time.time() * 1000)
            result    = landmarker.detect_for_video(mp_image, ts_ms)

            bbox        = None
            final_label = "uncertain"
            final_conf  = 0.0
            note        = ""
            yolo_label  = "no_face"

            face_detected = bool(result.face_landmarks)

            if face_detected:
                face_present_count += 1
                face_missing_count  = 0
                landmarks = result.face_landmarks[0]
                roi, bbox = crop_face(frame, landmarks)

                if roi is not None and roi.size > 0:
                    fw = bbox[2] - bbox[0]
                    fh = bbox[3] - bbox[1]
                    if fw < min_face_w or fh < min_face_h:
                        yolo_label = "too_far"
                        note       = "Move closer to camera"
                    else:
                        yolo_label, yolo_conf = classify_roi(cls_model, roi)
                        if yolo_conf < conf_thresh:
                            yolo_label = "uncertain"
                            yolo_conf  = 0.0
                        history.append((yolo_label, yolo_conf))

                        stable_lbl, stable_conf = stable_label(history, min_votes)

                        if in_missing_alert and face_present_count < reacquire_thresh:
                            final_label = "distracted"
                            final_conf  = 1.0
                            note        = "Recovering..."
                        else:
                            in_missing_alert = False
                            if yolo_label == "too_far":
                                final_label = "too_far"
                            elif stable_lbl == "drowsy":
                                drowsy_streak  += 1
                                yawning_streak  = 0
                                if drowsy_streak >= drowsy_hold:
                                    final_label        = "drowsy"
                                    final_conf         = stable_conf
                                    last_confirmed_lbl = "drowsy"
                                else:
                                    final_label = last_confirmed_lbl
                                    final_conf  = last_confirmed_conf
                                    note = f"Drowsy pending {drowsy_streak}/{drowsy_hold}"
                            elif stable_lbl == "yawning":
                                yawning_streak += 1
                                drowsy_streak   = 0
                                if yawning_streak >= yawning_hold:
                                    final_label        = "yawning"
                                    final_conf         = stable_conf
                                    last_confirmed_lbl = "yawning"
                                else:
                                    final_label = last_confirmed_lbl
                                    final_conf  = last_confirmed_conf
                                    note = f"Yawn pending {yawning_streak}/{yawning_hold}"
                            elif stable_lbl in REAL_LABELS:
                                drowsy_streak      = 0
                                yawning_streak     = 0
                                final_label        = stable_lbl
                                final_conf         = stable_conf
                                last_confirmed_lbl = stable_lbl
                                last_confirmed_conf = stable_conf
                            else:
                                final_label = "uncertain"
                else:
                    history.append(("uncertain", 0.0))
            else:
                face_present_count  = 0
                face_missing_count += 1
                drowsy_streak       = 0
                yawning_streak      = 0
                history.append(("no_face", 0.0))

                if face_missing_count >= missing_thresh:
                    in_missing_alert = True
                    final_label = "distracted"
                    final_conf  = 1.0
                    note        = "Face missing too long!"
                else:
                    final_label = "no_face"
                    note        = "No face detected"

            # Draw bbox
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr(final_label), 2)
                cv2.putText(frame, final_label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_bgr(final_label), 2, cv2.LINE_AA)

            draw_overlay(frame, final_label, final_conf, note)

            # FPS
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start   = time.time()
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            # Update global state
            is_alert = final_label in VISUAL_ALERT_LABELS
            with state_lock:
                detection_state["label"]      = final_label
                detection_state["confidence"] = round(final_conf, 3)
                detection_state["alert"]      = is_alert
                detection_state["note"]       = note
                detection_state["fps"]        = round(current_fps, 1)
                detection_state["color"]      = LABEL_COLORS_HEX.get(final_label, "#90a4ae")
                if is_alert:
                    history_list = detection_state["alert_history"]
                    ts = time.strftime("%H:%M:%S")
                    if not history_list or history_list[-1]["label"] != final_label:
                        history_list.append({"time": ts, "label": final_label})
                        if len(history_list) > 20:
                            history_list.pop(0)

            # Encode frame for MJPEG
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with output_frame_lock:
                output_frame = buf.tobytes()

    cap.release()
    with state_lock:
        detection_state["running"] = False
        detection_state["label"]   = "no_face"
        detection_state["alert"]   = False


# ──────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    # Load metrics for dashboard
    metrics = {}
    report_path = REPORTS_DIR / "classification_report_test.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            metrics = json.load(f)

    test_summary = {}
    summary_path = REPORTS_DIR / "test_metrics_summary.txt"
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                test_summary[k.strip()] = v.strip()

    train_summary = {}
    train_path = REPORTS_DIR / "train_metrics_summary.txt"
    if train_path.exists():
        for line in train_path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                train_summary[k.strip()] = v.strip()

    return render_template("index.html",
                           metrics=metrics,
                           test_summary=test_summary,
                           train_summary=train_summary)


@app.route("/results")
def results():
    figures = {}
    for name in ["accuracy_curve", "loss_curve", "confusion_matrix_test",
                 "classification_report_table_test", "per_class_metrics_test", "results_train"]:
        p = FIGURES_DIR / f"{name}.png"
        figures[name] = p.exists()

    report = {}
    report_path = REPORTS_DIR / "classification_report_test.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

    test_summary = {}
    summary_path = REPORTS_DIR / "test_metrics_summary.txt"
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                test_summary[k.strip()] = v.strip()

    return render_template("results.html", figures=figures,
                           report=report, test_summary=test_summary)


@app.route("/demo")
def demo():
    return render_template("demo.html")


# ── Static figure serving ──
@app.route("/figures/<name>")
def serve_figure(name: str):
    from flask import send_from_directory
    return send_from_directory(str(FIGURES_DIR), name)


# ── Camera control ──
@app.route("/api/start_camera", methods=["POST"])
def start_camera():
    global camera_thread, stop_event
    with state_lock:
        if detection_state["running"]:
            return jsonify({"ok": False, "msg": "Already running"})

    data = request.get_json(silent=True) or {}
    stop_event.clear()
    camera_thread = threading.Thread(
        target=camera_worker,
        kwargs={
            "camera_index": int(data.get("camera", 0)),
            "window":       int(data.get("window", 9)),
            "min_votes":    int(data.get("min_votes", 6)),
            "conf_thresh":  float(data.get("conf", 0.45)),
        },
        daemon=True,
    )
    with state_lock:
        detection_state["running"]       = True
        detection_state["alert_history"] = []
    camera_thread.start()
    return jsonify({"ok": True})


@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():
    stop_event.set()
    with state_lock:
        detection_state["running"] = False
        detection_state["label"]   = "no_face"
        detection_state["alert"]   = False
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(detection_state))


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with output_frame_lock:
                frame_bytes = output_frame
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)   # ~30 fps cap

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
