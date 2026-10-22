# src/track_wholly.py
"""
Whole-person tracking with face association and the honest three-state
identity model (CONFIRMED / TEMPORARY / UNKNOWN). No motor output - this is
the vision module; see face-recognition/src/track.py for the ESP32 pan
pattern this would feed into.

Run:
    python -m src.track_wholly --target "SomeEnrolledName" [--log runs/session.csv]

Keys:
    q : quit
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Optional, TextIO

import cv2

from .camera import pick_camera_index
from .face_pipeline import ArcFaceEmbedderONNX, FaceDBMatcher, MultiFaceDetector, load_db_npz
from .identity_state import IdentityState
from .person_detector import PersonDetector
from .whole_person_tracker import WholePersonTracker

logger = logging.getLogger(__name__)

STATE_COLOR = {
    IdentityState.CONFIRMED: (0, 200, 0),
    IdentityState.TEMPORARY: (0, 165, 255),
    IdentityState.UNKNOWN: (0, 0, 255),
    IdentityState.SEARCHING: (160, 160, 160),
}


def draw_label(frame, text, xy, color, scale=0.62):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def open_log(path: Optional[str]) -> Optional[TextIO]:
    if path is None:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", newline="")
    csv.writer(f).writerow(["t", "frame", "state", "confidence", "error_x", "error_y", "face_seen"])
    return f


def log_row(f: Optional[TextIO], frame_index: int, result) -> None:
    if f is None:
        return
    row = [f"{time.time():.3f}", frame_index]
    if result is None:
        row += ["SEARCHING", 0.0, "", "", 0]
    else:
        row += [result.identity.state.name, f"{result.identity.confidence:.3f}",
                f"{result.position.error_x:.3f}", f"{result.position.error_y:.3f}",
                int(result.face is not None)]
    csv.writer(f).writerow(row)
    f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="enrolled identity to lock onto")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.34, help="ArcFace cosine-distance accept threshold")
    parser.add_argument("--face-detect-conf", type=float, default=0.5, help="BlazeFace min detection confidence")
    parser.add_argument("--person-detect-conf", type=float, default=0.4, help="EfficientDet-Lite0 min person-detection confidence")
    parser.add_argument("--body-detect-every", type=int, default=2,
                         help="run the (expensive) person detector every N frames; continuity carries the box between")
    parser.add_argument("--appearance-threshold", type=float, default=0.55,
                         help="min torso-histogram correlation to re-acquire by appearance alone (e.g. back turned)")
    parser.add_argument("--decay-rate", type=float, default=0.35, help="per-second confidence decay while TEMPORARY")
    parser.add_argument("--temporary-floor", type=float, default=0.35, help="confidence below this -> UNKNOWN")
    parser.add_argument("--log", default=None, help="CSV path to record state/confidence per frame for evaluation")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-mirror", action="store_true",
                         help="disable the default mirror flip (by default the view is flipped "
                              "horizontally so your right appears on your right, like a mirror)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    face_detector = MultiFaceDetector(min_confidence=args.face_detect_conf)
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(load_db_npz(Path("data/db/face_db.npz")), dist_thresh=args.threshold)
    body_detector = PersonDetector(min_confidence=args.person_detect_conf)

    appearance_db = load_db_npz(Path("data/db/appearance_db.npz"))
    appearance_signature = appearance_db.get(args.target)
    if appearance_signature is None:
        logger.warning("no appearance signature enrolled for %r - back-turned/occluded re-acquisition "
                        "will fall back to face-only (see src/enroll.py)", args.target)

    tracker = WholePersonTracker(args.target, face_detector, embedder, matcher, body_detector,
                                  body_detect_every=args.body_detect_every,
                                  appearance_signature=appearance_signature,
                                  appearance_threshold=args.appearance_threshold)
    tracker.identity.decay_rate = args.decay_rate
    tracker.identity.temporary_floor = args.temporary_floor

    camera_index = args.camera if args.camera is not None else pick_camera_index()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {camera_index} not available.")

    log_file = open_log(args.log)
    logger.info("track_wholly started: target=%r camera=%d", args.target, camera_index)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            result = tracker.update(frame)
            log_row(log_file, tracker.frame_index, result)
            view = frame.copy()

            if result is None:
                draw_label(view, f"SEARCHING for {args.target}", (12, 28), STATE_COLOR[IdentityState.SEARCHING], 0.72)
            else:
                color = STATE_COLOR[result.identity.state]
                b = result.body
                cv2.rectangle(view, (b.x1, b.y1), (b.x2, b.y2), color, 3)
                if result.face is not None:
                    f = result.face
                    cv2.rectangle(view, (f.x1, f.y1), (f.x2, f.y2), color, 2)

                draw_label(view,
                           f"{result.identity.state.name}: {args.target}  conf={result.identity.confidence:.2f}",
                           (12, 28), color, 0.72)
                draw_label(view,
                           f"H={result.position.horizontal} V={result.position.vertical} "
                           f"error=({result.position.error_x:+.2f},{result.position.error_y:+.2f}) "
                           f"since_confirmed={result.identity.seconds_since_confirmed:.1f}s",
                           (12, 56), color, 0.55)

            h, w = view.shape[:2]
            dz = tracker.dead_zone
            cv2.rectangle(view,
                          (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                          (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                          (120, 120, 120), 1)
            cv2.imshow("Whole-Person Tracking", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        face_detector.close()
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
