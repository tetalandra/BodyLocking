# src/face_pipeline.py
"""
Multi-face detect + embed + match stack:
MediaPipe FaceDetector (BlazeFace, multi-face) -> MediaPipe FaceLandmarker 5pt
(per-face ROI) -> align_face_5pt (112x112) -> ArcFace ONNX embedding -> cosine
distance to DB.

This is the identity-verification half of whole-person tracking (src/whole_person_tracker.py);
the body-detection half lives in src/person_detector.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

from .align import align_face_5pt
from .face_detector import FaceBoxDetector, _DEFAULT_MODEL_PATH as _DEFAULT_FACE_DETECTOR_MODEL

_DEFAULT_LANDMARKER_MODEL = Path(__file__).parent.parent / "models" / "face_landmarker.task"


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray  # (5,2) float32, full-frame coords


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1).astype(np.float32), b.reshape(-1).astype(np.float32)))


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _bbox_from_5pt(kps: np.ndarray, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min, x_max = float(k[:, 0].min()), float(k[:, 0].max())
    y_min, y_max = float(k[:, 1].min()), float(k[:, 1].max())
    w, h = max(1.0, x_max - x_min), max(1.0, y_max - y_min)
    return np.array(
        [x_min - pad_x * w, y_min - pad_y_top * h, x_max + pad_x * w, y_max + pad_y_bot * h],
        dtype=np.float32,
    )


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists() or db_path.stat().st_size == 0:
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}


class ArcFaceEmbedderONNX:
    """ArcFace-style ONNX embedder. Input 112x112 BGR -> L2-normalized (D,) embedding."""

    def __init__(self, model_path: str = "models/embedder_arcface.onnx", input_size=(112, 112)):
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        return np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        y = self.sess.run([self.out_name], {self.in_name: self._preprocess(aligned_bgr_112)})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return (emb / (np.linalg.norm(emb) + 1e-12)).astype(np.float32)


class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self) -> None:
        self._names = sorted(self.db.keys())
        self._mat = (
            np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
            if self._names else None
        )

    def reload_from(self, path: Path) -> None:
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        sims = (self._mat @ emb.reshape(1, -1).astype(np.float32).T).reshape(-1)
        best_i = int(np.argmax(sims))
        best_dist = 1.0 - float(sims[best_i])
        ok = best_dist <= self.dist_thresh
        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=best_dist,
            similarity=float(sims[best_i]),
            accepted=ok,
        )


class MultiFaceDetector:
    """Multi-face: MediaPipe FaceDetector (BlazeFace) -> per-face-ROI MediaPipe FaceLandmarker 5pt."""

    def __init__(
        self,
        face_detector_model: str = _DEFAULT_FACE_DETECTOR_MODEL,
        landmarker_model: str = str(_DEFAULT_LANDMARKER_MODEL),
        min_confidence: float = 0.5,
        max_dim: int = 640,
    ):
        self.box_detector = FaceBoxDetector(model_path=face_detector_model, min_confidence=min_confidence, max_dim=max_dim)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=landmarker_model),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            # lower than the box detector's own threshold - a moderately pitched-down
            # face (e.g. looking down at a phone) still passes the box detector but
            # was failing the landmarker's own internal face check, dropping the face
            # entirely out of detect() and triggering an unnecessary CONFIRMED->TEMPORARY
            # drop on every re-verification that happened to land during the glance
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.IDX = (33, 263, 1, 61, 291)  # left eye, right eye, nose, mouth left, mouth right

    def close(self) -> None:
        self.box_detector.close()
        self.landmarker.close()

    def _roi_landmarks_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect(mp_image)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        kps = np.array([[lm[i].x * W, lm[i].y * H] for i in self.IDX], dtype=np.float32)
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]
        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 8) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        faces = self.box_detector.detect(frame_bgr, max_faces=max_faces)
        if not faces:
            return []

        out: List[FaceDet] = []
        for face in faces:
            w, h = face.x2 - face.x1, face.y2 - face.y1
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(face.x1 - mx, face.y1 - my, face.x2 + mx, face.y2 + my, W, H)
            kps_roi = self._roi_landmarks_5pt(frame_bgr[ry1:ry2, rx1:rx2])
            if kps_roi is None:
                continue
            kps = kps_roi + np.array([rx1, ry1], dtype=np.float32)

            eye_dist = float(np.linalg.norm(kps[1] - kps[0]))
            if eye_dist < max(10.0, 0.18 * float(w)):
                continue

            bb = _bbox_from_5pt(kps)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)
            out.append(FaceDet(x1=x1, y1=y1, x2=x2, y2=y2, kps=kps.astype(np.float32)))
        return out


def identity_of(embedder: ArcFaceEmbedderONNX, matcher: FaceDBMatcher, frame: np.ndarray, face: FaceDet) -> MatchResult:
    aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
    if aligned is None:
        return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
    return matcher.match(embedder.embed(aligned))
