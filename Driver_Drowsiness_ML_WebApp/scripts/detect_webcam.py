from __future__ import annotations

import argparse
import platform
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

# MediaPipe Tasks imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VISUAL_ALERT_LABELS = {"drowsy", "yawning", "looking_away", "distracted"}
SOUND_ALERT_LABELS = {"drowsy", "yawning", "distracted"}
REAL_LABELS = {"awake", "drowsy", "yawning", "looking_away"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime driver state detection with MediaPipe Tasks + YOLO classification + missing-face fallback."
    )
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--face-model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--camera", type=int, default=0)

    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.45)

    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--min-votes", type=int, default=6)

    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--draw-landmarks", action="store_true")

    # Too-far filtering
    parser.add_argument("--min-face-w", type=int, default=140)
    parser.add_argument("--min-face-h", type=int, default=140)

    # Stability / label hold
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=0,
        help="Hold last confirmed label for a few frames when current prediction becomes uncertain.",
    )
    parser.add_argument(
        "--drowsy-hold-frames",
        type=int,
        default=10,
        help="Require drowsy to persist for N consecutive frames before confirming it.",
    )
    parser.add_argument(
        "--yawning-hold-frames",
        type=int,
        default=12,
        help="Require yawning to persist for N consecutive frames before confirming it.",
    )

    # Fallback logic
    parser.add_argument(
        "--missing-threshold",
        type=int,
        default=10,
        help="Number of consecutive frames without a detected face before raising distracted alert.",
    )
    parser.add_argument(
        "--reacquire-threshold",
        type=int,
        default=2,
        help="Number of consecutive detected-face frames to recover from distracted state.",
    )
    parser.add_argument(
        "--show-topk",
        action="store_true",
        help="Show top predictions on terminal for debugging.",
    )

    # Overlay / debug
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Show extra debug info on frame (YOLO raw label, face size, counters).",
    )

    # Beep
    parser.add_argument("--beep", action="store_true", help="Enable sound alert.")
    parser.add_argument(
        "--beep-freq",
        type=int,
        default=1800,
        help="Beep frequency in Hz (Windows only).",
    )
    parser.add_argument(
        "--beep-duration",
        type=int,
        default=350,
        help="Beep duration in milliseconds (Windows only).",
    )
    parser.add_argument(
        "--beep-cooldown",
        type=float,
        default=1.2,
        help="Minimum seconds between two alert beeps.",
    )

    return parser.parse_args()


def classify_roi(model: YOLO, roi_bgr, imgsz: int):
    results = model.predict(source=roi_bgr, imgsz=imgsz, verbose=False)
    if not results:
        return "uncertain", 0.0, []

    res = results[0]
    probs = getattr(res, "probs", None)
    if probs is None:
        return "uncertain", 0.0, []

    top1_idx = int(probs.top1)
    conf = float(probs.top1conf)
    label = res.names[top1_idx]

    top5 = probs.top5
    top5conf = probs.top5conf.tolist()
    pairs = [(res.names[int(i)], float(c)) for i, c in zip(top5, top5conf)]

    return label, conf, pairs


def stable_label(history: deque[tuple[str, float]], min_votes: int) -> tuple[str, float]:
    valid = [label for label, _ in history if label in REAL_LABELS]
    if not valid:
        return "uncertain", 0.0

    counts = Counter(valid)
    label, votes = counts.most_common(1)[0]
    if votes < min_votes:
        return "uncertain", 0.0

    confs = [conf for hist_label, conf in history if hist_label == label]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return label, avg_conf


def color_for_label(label: str) -> tuple[int, int, int]:
    return {
        "awake": (0, 200, 0),
        "drowsy": (0, 0, 255),
        "yawning": (0, 165, 255),
        "looking_away": (255, 128, 0),
        "distracted": (255, 0, 255),
        "too_far": (0, 215, 255),
        "uncertain": (180, 180, 180),
        "no_face": (120, 120, 120),
    }.get(label, (255, 255, 255))


def crop_face_from_landmarks(frame_bgr, landmarks, margin: float = 0.18):
    h, w = frame_bgr.shape[:2]

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min = max(0.0, min(xs) - margin)
    y_min = max(0.0, min(ys) - margin)
    x_max = min(1.0, max(xs) + margin)
    y_max = min(1.0, max(ys) + margin)

    x1 = int(x_min * w)
    y1 = int(y_min * h)
    x2 = int(x_max * w)
    y2 = int(y_max * h)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame_bgr[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def draw_face_landmarks(frame_bgr, landmarks):
    h, w = frame_bgr.shape[:2]
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), 1, (255, 255, 0), -1)


def play_alert_beep(freq: int = 1800, duration: int = 350):
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(freq, duration)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def draw_translucent_panel(frame, x: int, y: int, w: int, h: int, alpha: float = 0.45):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_status_panel(
    frame,
    final_label: str,
    final_conf: float,
    note: str | None = None,
    yolo_label: str | None = None,
    yolo_conf: float | None = None,
    face_present_count: int | None = None,
    face_missing_count: int | None = None,
    face_w: int | None = None,
    face_h: int | None = None,
    debug_overlay: bool = False,
):
    color = color_for_label(final_label)

    panel_w = 340 if debug_overlay else 260
    panel_h = 175 if debug_overlay else 95
    draw_translucent_panel(frame, 12, 12, panel_w, panel_h, alpha=0.55)

    cv2.putText(
        frame,
        "Driver State",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"{final_label.upper()}",
        (24, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Conf: {final_conf:.2f}",
        (24, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    if note:
        cv2.putText(
            frame,
            note,
            (24, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    if debug_overlay:
        y = 160
        if yolo_label is not None and yolo_conf is not None:
            cv2.putText(
                frame,
                f"YOLO: {yolo_label} ({yolo_conf:.2f})",
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            y += 24

        if face_w is not None and face_h is not None:
            cv2.putText(
                frame,
                f"Face size: {face_w}x{face_h}",
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            y += 24

        if face_present_count is not None and face_missing_count is not None:
            cv2.putText(
                frame,
                f"Present: {face_present_count} | Missing: {face_missing_count}",
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )

    if final_label in VISUAL_ALERT_LABELS:
        cv2.putText(
            frame,
            "ALERT!",
            (24, panel_h + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.face_model.exists():
        raise FileNotFoundError(
            f"Face Landmarker model not found: {args.face_model}\n"
            f"Download a compatible .task model and place it at this path."
        )

    cls_model = YOLO(str(args.weights))

    BaseOptions = python.BaseOptions
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.face_model)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    history = deque(maxlen=args.window)

    face_missing_count = 0
    face_present_count = 0
    in_missing_alert = False
    last_beep_time = 0.0

    last_confirmed_label = "awake"
    last_confirmed_conf = 0.0
    hold_counter = 0
    drowsy_streak = 0
    yawning_streak = 0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    with FaceLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[INFO] Failed to read frame. Exiting.")
                    break

                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                bbox = None
                face_w = None
                face_h = None
                yolo_label = "no_face"
                yolo_conf = 0.0
                final_label = "uncertain"
                final_conf = 0.0
                note = None

                face_detected = bool(result.face_landmarks)

                if face_detected:
                    face_present_count += 1
                    face_missing_count = 0

                    landmarks = result.face_landmarks[0]
                    roi, bbox = crop_face_from_landmarks(frame, landmarks, margin=args.margin)

                    if roi is not None and roi.size > 0 and bbox is not None:
                        x1, y1, x2, y2 = bbox
                        face_w = x2 - x1
                        face_h = y2 - y1

                        if face_w < args.min_face_w or face_h < args.min_face_h:
                            yolo_label = "too_far"
                            yolo_conf = 0.0
                            note = "Move closer"
                        else:
                            yolo_label, yolo_conf, top_preds = classify_roi(cls_model, roi, imgsz=args.imgsz)

                            if args.show_topk:
                                print(top_preds[:4])

                            if yolo_conf < args.conf:
                                yolo_label = "uncertain"
                                yolo_conf = 0.0
                    else:
                        yolo_label = "uncertain"
                        yolo_conf = 0.0

                    history.append((yolo_label, yolo_conf))
                    stable_yolo_label, stable_yolo_conf = stable_label(history, args.min_votes)

                    if in_missing_alert and face_present_count < args.reacquire_threshold:
                        drowsy_streak = 0
                        yawning_streak = 0
                        final_label = "distracted"
                        final_conf = 1.0
                        note = "Face missing too long"
                    else:
                        in_missing_alert = False

                        if yolo_label == "too_far":
                            drowsy_streak = 0
                            yawning_streak = 0
                            final_label = "too_far"
                            final_conf = 0.0

                        elif stable_yolo_label == "drowsy":
                            drowsy_streak += 1
                            yawning_streak = 0

                            if drowsy_streak >= args.drowsy_hold_frames:
                                final_label = "drowsy"
                                final_conf = stable_yolo_conf
                                last_confirmed_label = "drowsy"
                                last_confirmed_conf = stable_yolo_conf
                                hold_counter = 0
                            else:
                                if last_confirmed_label in REAL_LABELS and last_confirmed_label != "drowsy":
                                    final_label = last_confirmed_label
                                    final_conf = last_confirmed_conf
                                else:
                                    final_label = "uncertain"
                                    final_conf = 0.0
                                note = f"Drowsy pending {drowsy_streak}/{args.drowsy_hold_frames}"

                        elif stable_yolo_label == "yawning":
                            yawning_streak += 1
                            drowsy_streak = 0

                            if yawning_streak >= args.yawning_hold_frames:
                                final_label = "yawning"
                                final_conf = stable_yolo_conf
                                last_confirmed_label = "yawning"
                                last_confirmed_conf = stable_yolo_conf
                                hold_counter = args.hold_frames
                            else:
                                if last_confirmed_label in REAL_LABELS and last_confirmed_label != "yawning":
                                    final_label = last_confirmed_label
                                    final_conf = last_confirmed_conf
                                else:
                                    final_label = "uncertain"
                                    final_conf = 0.0
                                note = f"Yawn pending {yawning_streak}/{args.yawning_hold_frames}"

                        elif stable_yolo_label in REAL_LABELS:
                            drowsy_streak = 0
                            yawning_streak = 0
                            final_label = stable_yolo_label
                            final_conf = stable_yolo_conf
                            last_confirmed_label = stable_yolo_label
                            last_confirmed_conf = stable_yolo_conf
                            hold_counter = args.hold_frames

                        elif hold_counter > 0 and last_confirmed_label != "drowsy":
                            drowsy_streak = 0
                            yawning_streak = 0
                            final_label = last_confirmed_label
                            final_conf = last_confirmed_conf
                            hold_counter -= 1
                            note = "Stabilizing..."

                        else:
                            drowsy_streak = 0
                            yawning_streak = 0
                            final_label = "uncertain"
                            final_conf = 0.0

                    if args.draw_landmarks:
                        draw_face_landmarks(frame, landmarks)

                else:
                    face_present_count = 0
                    face_missing_count += 1
                    drowsy_streak = 0
                    yawning_streak = 0
                    history.append(("no_face", 0.0))

                    if face_missing_count >= args.missing_threshold:
                        final_label = "distracted"
                        final_conf = 1.0
                        note = "Face missing too long"
                    else:
                        final_label = "no_face"
                        final_conf = 0.0
                        note = "No face detected"

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    color = color_for_label(final_label)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(
                        frame,
                        f"{final_label}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                draw_status_panel(
                    frame=frame,
                    final_label=final_label,
                    final_conf=final_conf,
                    note=note,
                    yolo_label=yolo_label,
                    yolo_conf=yolo_conf,
                    face_present_count=face_present_count,
                    face_missing_count=face_missing_count,
                    face_w=face_w,
                    face_h=face_h,
                    debug_overlay=args.debug_overlay,
                )

                now = time.time()
                if (
                    args.beep
                    and face_detected
                    and final_label in SOUND_ALERT_LABELS
                    and (now - last_beep_time) >= args.beep_cooldown
                ):
                    play_alert_beep(args.beep_freq, args.beep_duration)
                    last_beep_time = now

                cv2.imshow("Driver Drowsiness Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()