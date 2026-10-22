# src/enroll.py
"""
Enrollment: guides the person through FRONT / LEFT / BACK / RIGHT turns so
the target is recognizable from any orientation, not just facing the
camera - a face embedding alone is useless once someone turns away, which
is exactly the gap src/appearance.py exists to close for tracking.

Per phase: camera -> MultiFaceDetector (face embedding, when a face is
visible - mainly FRONT) and PersonDetector (torso appearance signature,
every phase including BACK, since clothing doesn't need a face). Saves
the mean face embedding to data/db/face_db.npz and the mean appearance
signature (averaged across all four orientations, so it generalizes) to
data/db/appearance_db.npz.

Run:
    python -m src.enroll --name Civil
    python -m src.enroll --name Civil --phase-seconds 5   # slower, e.g. for testing

Shows a live window with on-screen rotation prompts; 'q' aborts. Falls
back to console-only progress if no display is available.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .align import align_face_5pt
from .appearance import torso_signature
from .camera import pick_camera_index
from .face_pipeline import ArcFaceEmbedderONNX, MultiFaceDetector, load_db_npz
from .person_detector import PersonDetector

FACE_DB_PATH = Path("data/db/face_db.npz")
APPEARANCE_DB_PATH = Path("data/db/appearance_db.npz")
SAMPLES_NEEDED = 10

PHASES: List[Tuple[str, str]] = [
    ("FRONT", "Face the camera"),
    ("LEFT", "Turn to show your LEFT side"),
    ("BACK", "Turn your BACK to the camera"),
    ("RIGHT", "Turn to show your RIGHT side"),
]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    return _normalize(np.stack(embeddings, axis=0).mean(axis=0))


def mean_signature(signatures: List[np.ndarray]) -> np.ndarray:
    m = np.stack(signatures, axis=0).mean(axis=0)
    s = m.sum()
    return (m / s if s > 0 else m).astype(np.float32)


def save_db(path: Path, db: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v.astype(np.float32) for k, v in db.items()})


def _try_imshow(window: str, frame: np.ndarray) -> bool:
    try:
        cv2.imshow(window, frame)
        return True
    except cv2.error:
        return False


def run_enrollment(name: str, phase_seconds: float = 3.5, capture_every: float = 0.2, mirror: bool = True) -> None:
    face_detector = MultiFaceDetector()
    body_detector = PersonDetector()
    embedder = ArcFaceEmbedderONNX()

    face_db = load_db_npz(FACE_DB_PATH)
    appearance_db = load_db_npz(APPEARANCE_DB_PATH)
    face_samples: List[np.ndarray] = []
    appearance_samples: List[np.ndarray] = []

    cap = cv2.VideoCapture(pick_camera_index())
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    window_available = True
    aborted = False
    try:
        for phase_name, instruction in PHASES:
            print(f"[{phase_name}] {instruction} ({phase_seconds:.0f}s)")
            last_capture = 0.0
            deadline = time.time() + phase_seconds
            phase_faces = phase_bodies = 0

            while time.time() < deadline:
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                faces = face_detector.detect(frame, max_faces=1)
                bodies = body_detector.detect(frame, max_people=1)
                now = time.time()

                if now - last_capture >= capture_every:
                    if faces:
                        aligned, _ = align_face_5pt(frame, faces[0].kps, out_size=(112, 112))
                        if aligned is not None:
                            face_samples.append(embedder.embed(aligned))
                            phase_faces += 1
                    if bodies:
                        b = bodies[0]
                        appearance_samples.append(torso_signature(frame, (b.x1, b.y1, b.x2, b.y2)))
                        phase_bodies += 1
                    last_capture = now

                if window_available:
                    vis = frame.copy()
                    if faces:
                        f = faces[0]
                        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                    if bodies:
                        b = bodies[0]
                        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (255, 170, 0), 2)
                    remaining = max(0.0, deadline - time.time())
                    cv2.putText(vis, f"{name}: {phase_name} - {instruction}", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.putText(vis, f"{remaining:.1f}s  faces={len(face_samples)} appearance={len(appearance_samples)}",
                                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
                    window_available = _try_imshow("enroll", vis)
                    if window_available and (cv2.waitKey(1) & 0xFF) == ord("q"):
                        aborted = True
                        break

            print(f"  captured {phase_faces} face samples, {phase_bodies} appearance samples this phase")
            if aborted:
                break
    finally:
        face_detector.close()
        body_detector.close()
        cap.release()
        if window_available:
            cv2.destroyAllWindows()

    if aborted:
        print("aborted, nothing saved.")
        return

    print(f"total: {len(face_samples)} face samples, {len(appearance_samples)} appearance samples")
    if len(face_samples) < SAMPLES_NEEDED:
        print(f"not enough face samples ({len(face_samples)}/{SAMPLES_NEEDED}) - make sure you were "
              f"facing the camera during the FRONT phase and well lit, then try again.")
        return
    if len(appearance_samples) < SAMPLES_NEEDED:
        print(f"not enough appearance samples ({len(appearance_samples)}/{SAMPLES_NEEDED}) - make sure "
              f"your whole body was in frame, then try again.")
        return

    face_db[name] = mean_embedding(face_samples)
    appearance_db[name] = mean_signature(appearance_samples)
    save_db(FACE_DB_PATH, face_db)
    save_db(APPEARANCE_DB_PATH, appearance_db)
    print(f"saved '{name}': {FACE_DB_PATH} ({len(face_db)} identities), "
          f"{APPEARANCE_DB_PATH} ({len(appearance_db)} identities)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, help="identity to enroll; prompted if omitted")
    parser.add_argument("--phase-seconds", type=float, default=3.5,
                         help="seconds to hold each of the 4 rotation phases (FRONT/LEFT/BACK/RIGHT)")
    parser.add_argument("--no-mirror", action="store_true",
                         help="disable the default mirror flip - must match whatever src.track_wholly "
                              "uses, since face embeddings aren't flip-invariant")
    args = parser.parse_args()

    name = args.name or input("Enter person name to enroll (e.g., Civil): ").strip()
    if not name:
        print("No name provided. Exiting.")
        return
    run_enrollment(name, phase_seconds=args.phase_seconds, mirror=not args.no_mirror)


if __name__ == "__main__":
    main()
