import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set

# ─────────────────────────────────────────────────────────────────────────────
# FaceTracker: Hybrid Optical Flow + Ghost Blur + Session Identity Cache
#
# Identity Rules:
#   - New/unknown face  → NOT blurred until MobileFaceNet confirms non-vlogger
#   - Confirmed vlogger → NEVER blurred, even when re-entering the frame
#   - Confirmed non-vlogger → blurred IMMEDIATELY on re-detection via cache
# ─────────────────────────────────────────────────────────────────────────────

GHOST_FRAMES   = 10   # Frames to keep track active after last AI detection
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

class TrackedFace:
    def __init__(self, tid: int, box: List[int], is_vlogger: bool,
                 identity_confirmed: bool, frame_gray: np.ndarray):
        self.tid = tid
        self.box = box                          # [x1, y1, x2, y2]
        self.is_vlogger = is_vlogger
        self.identity_confirmed = identity_confirmed
        self.disappeared = 0  # Number of frames since last AI detection match

        # Seed optical flow points from the centre region of the face
        self.pts = self._seed_points(box, frame_gray)

    def _seed_points(self, box: List[int], gray: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = box
        roi = gray[max(0, y1):min(gray.shape[0], y2), max(0, x1):min(gray.shape[1], x2)]
        if roi.size == 0:
            return None
        # Good features to track inside the face ROI
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=20, qualityLevel=0.2,
                                      minDistance=5, blockSize=5)
        if pts is None or len(pts) == 0:
            # Fallback: use a grid of points
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            pts = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        else:
            # Shift back from ROI-local to full-frame coords
            pts[:, :, 0] += x1
            pts[:, :, 1] += y1
        return pts


class FaceTracker:
    def __init__(self, max_disappeared: int = GHOST_FRAMES):
        self.next_id: int = 0
        self.faces: Dict[int, TrackedFace] = {}
        self.max_disappeared = max_disappeared
        self.prev_gray: Optional[np.ndarray] = None
        self.session_cache: Dict[int, bool] = {}
        self.confirmed_non_vlogger_boxes: List[List[int]] = []

    def tick(self, frame: np.ndarray) -> List[Dict]:
        """
        Call every frame BETWEEN detection cycles.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None and self.faces:
            self._optical_flow_step(gray)
            # Age all tracks every frame. Only AI-detection re-match resets this.
            for tid in list(self.faces.keys()):
                self.faces[tid].disappeared += 1
                if self.faces[tid].disappeared > self.max_disappeared:
                    del self.faces[tid]

        self.prev_gray = gray
        return self.get_metadata()

    def reinit(self, frame: np.ndarray,
               detected_boxes: List[List[int]],
               vlogger_indices: List[int] = None,
               recognition_ran: bool = False) -> List[Dict]:
        """
        Call on detection frames.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vlogger_set: Set[int] = set(vlogger_indices or [])

        if not detected_boxes:
            self._age_disappeared()
            self.prev_gray = gray
            return self.get_metadata()

        if not self.faces:
            for i, box in enumerate(detected_boxes):
                is_v, confirmed = self._resolve_identity(
                    i, box, vlogger_set, recognition_ran, is_new=True
                )
                tf = TrackedFace(self.next_id, box, is_v, confirmed, gray)
                self.faces[self.next_id] = tf
                self.next_id += 1
            self.prev_gray = gray
            return self.get_metadata()

        # Match new detections to existing tracks
        t_ids = list(self.faces.keys())
        t_cents = [self._centroid(self.faces[tid].box) for tid in t_ids]
        d_cents = [self._centroid(b) for b in detected_boxes]

        dist = np.linalg.norm(
            np.array(t_cents)[:, np.newaxis] - np.array(d_cents), axis=2
        )

        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist[row, col] > 160:
                continue

            tid = t_ids[row]
            tf = self.faces[tid]
            tf.box = detected_boxes[col]
            tf.disappeared = 0  # <--- RESET ONLY ON AI DETECTION MATCH
            tf.pts = tf._seed_points(tf.box, gray)

            if recognition_ran:
                is_v = col in vlogger_set
                tf.is_vlogger = is_v
                tf.identity_confirmed = True
                self.session_cache[tid] = is_v
                if not is_v:
                    self.confirmed_non_vlogger_boxes.append(list(tf.box))
                    self.confirmed_non_vlogger_boxes = self.confirmed_non_vlogger_boxes[-20:]

            used_rows.add(row), used_cols.add(col)

        # Age unmatched tracks
        for row in set(range(len(t_ids))) - used_rows:
            tid = t_ids[row]
            self.faces[tid].disappeared += 1
            if self.faces[tid].disappeared > self.max_disappeared:
                del self.faces[tid]

        # Register unmatched new detections
        for col in set(range(len(detected_boxes))) - used_cols:
            is_v, confirmed = self._resolve_identity(
                col, detected_boxes[col], vlogger_set, recognition_ran, is_new=True
            )
            tf = TrackedFace(self.next_id, detected_boxes[col], is_v, confirmed, gray)
            self.faces[self.next_id] = tf
            self.next_id += 1

        self.prev_gray = gray
        return self.get_metadata()

    def get_metadata(self) -> List[Dict]:
        return [
            {
                "id": tid,
                "x": int(tf.box[0]), "y": int(tf.box[1]),
                "w": int(tf.box[2] - tf.box[0]),
                "h": int(tf.box[3] - tf.box[1]),
                "isVlogger": tf.is_vlogger,
                "confirmed": tf.identity_confirmed,
            }
            for tid, tf in self.faces.items()
        ]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _optical_flow_step(self, gray: np.ndarray):
        """Move all tracked face boxes using sparse LK optical flow."""
        for tid, tf in list(self.faces.items()):
            if tf.pts is None or len(tf.pts) == 0:
                continue

            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, tf.pts, None, **LK_PARAMS
            )

            if new_pts is None or status is None:
                continue

            good = status.ravel() == 1
            if good.sum() == 0:
                continue

            good_new = new_pts[good]
            good_old = tf.pts[good]

            dx = int(np.mean(good_new[:, 0, 0] - good_old[:, 0, 0]))
            dy = int(np.mean(good_new[:, 0, 1] - good_old[:, 0, 1]))

            fh, fw = gray.shape[:2]
            x1 = max(0, tf.box[0] + dx)
            y1 = max(0, tf.box[1] + dy)
            x2 = min(fw, tf.box[2] + dx)
            y2 = min(fh, tf.box[3] + dy)
            tf.box = [x1, y1, x2, y2]
            tf.pts = good_new.reshape(-1, 1, 2)
            # DO NOT reset disappeared here

    def _age_disappeared(self):
        for tid in list(self.faces.keys()):
            self.faces[tid].disappeared += 1
            if self.faces[tid].disappeared > self.max_disappeared:
                del self.faces[tid]

    def _resolve_identity(self, det_idx: int, box: List[int],
                          vlogger_set: Set[int], recognition_ran: bool,
                          is_new: bool) -> Tuple[bool, bool]:
        """
        Rules:
        - Recognition ran → use result directly.
        - Recognition didn't run → check session cache via spatial proximity.
        - Brand new face, no cache hit → default NOT blurred (wait for ID).
        """
        if recognition_ran:
            is_v = det_idx in vlogger_set
            return is_v, True

        # Check if this detection is close to a previously confirmed non-vlogger
        cent = self._centroid(box)
        for cached_box in self.confirmed_non_vlogger_boxes:
            c2 = self._centroid(cached_box)
            if np.linalg.norm(np.array(cent) - np.array(c2)) < 150:
                return False, True  # Non-vlogger cache hit → blur immediately

        # Unknown and unconfirmed → do NOT blur (protect vlogger on re-entry)
        return True, False  # isVlogger=True means "don't blur yet"

    def _centroid(self, box: List[int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
