# src/appearance.py
"""
Appearance (clothing/body) signature: a color histogram of the torso
region of a body box, used to re-identify the target by what they're
wearing when no face is available to check - someone's back turned, a
brief full occlusion, or just not due for an ArcFace re-verification.

This is deliberately weaker evidence than a face match (same clothing can
repeat across people, lighting shifts the histogram), so it only ever
restores WholePersonTracker to TEMPORARY, never CONFIRMED - a face match
is still required for that. See src/whole_person_tracker.py.
"""

from __future__ import annotations

import cv2
import numpy as np

# torso band: below the head, above where legs/background dominate - robust
# to the person's facing direction, since clothing color doesn't depend on
# which way they're turned. Kept low (starts past the chest) so a hand/arm
# raised to cover the face - which reaches down into the upper-chest area -
# doesn't inject skin-tone pixels into the sampled region.
_TORSO_Y_RANGE = (0.40, 0.75)
_TORSO_X_MARGIN = 0.15
_HIST_BINS = (12, 12, 12)  # H, S, V


def torso_signature(frame: np.ndarray, box) -> np.ndarray:
    """HSV histogram of the box's torso band, L1-normalized. box = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    ty1 = int(y1 + _TORSO_Y_RANGE[0] * h)
    ty2 = int(y1 + _TORSO_Y_RANGE[1] * h)
    tx1 = int(x1 + _TORSO_X_MARGIN * w)
    tx2 = int(x2 - _TORSO_X_MARGIN * w)

    H, W = frame.shape[:2]
    ty1, ty2 = max(0, ty1), min(H, ty2)
    tx1, tx2 = max(0, tx1), min(W, tx2)
    roi = frame[ty1:ty2, tx1:tx2]
    if roi.size == 0:
        return np.zeros(np.prod(_HIST_BINS), dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(_HIST_BINS),
                         [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)


def compare_signatures(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram correlation in [-1, 1]; 1.0 = identical distribution."""
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
