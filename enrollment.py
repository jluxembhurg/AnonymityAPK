import cv2
import numpy as np
import os
import time
from detector import FaceDetector
from recognizer import FaceRecognizer

class EnrollmentModule:
    def __init__(self, detector=None, recognizer=None):
        self.detector = detector if detector else FaceDetector()
        self.recognizer = recognizer if recognizer else FaceRecognizer()
        self.save_path = "data/vlogger_profiles.npy"
        
        # 9 Zones: Center, North, NE, East, SE, South, SW, West, NW
        self.zones = ["C", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        self.coverage = {z: 0 for z in self.zones}
        self.target_per_zone = 3
        self.embeddings = []
        self.frame_count = 0
        
        self.compass_map = {
            "C":  (0, 0), "N":  (0, -1), "NE": (1, -1), "E":  (1, 0),
            "SE": (1, 1), "S":  (0, 1), "SW": (-1, 1), "W":  (-1, 0), "NW": (-1, -1)
        }
        self.instr_map = {
            "N": "Look UP", "S": "Look DOWN", 
            "E": "Look RIGHT", "W": "Look LEFT",
            "NE": "Look UP-RIGHT", "NW": "Look UP-LEFT", 
            "SE": "Look DOWN-RIGHT", "SW": "Look DOWN-LEFT"
        }

    def reset(self):
        self.embeddings = []
        self.coverage = {z: 0 for z in self.zones}
        self.frame_count = 0
        print("[ENROLL] Modules Reset")

    def get_coverage_zone(self, norm_x: float, norm_y: float) -> str:
        dx = norm_x - 0.5
        dy = norm_y - 0.5
        
        # Proportional Screen-Safe Grid
        # h=0.02 and v=0.035 map inversely to 16:9 to create a PERFECT SQUARE UI grid
        # Center cell is `w-4%`, `h-7%`. Total 3x3 Grid is `w-12%`, `h-21%`.
        h_thresh = 0.02 
        v_thresh = 0.035 
        diag_thresh = 0.02 # Strict square diagonal locking
        
        if abs(dx) < h_thresh and abs(dy) < v_thresh: 
            return "C"
        
        if dy < -v_thresh: # North area
            if dx < -diag_thresh: return "NW"
            if dx > diag_thresh: return "NE"
            return "N"
        if dy > v_thresh: # South area
            if dx < -diag_thresh: return "SW"
            if dx > diag_thresh: return "SE"
            return "S"
            
        # Middle vertical band (E/W)
        if dx < -h_thresh: return "W"
        if dx > h_thresh: return "E"
        return "C"

    def process_enrollment_frame(self, frame):
        """Processes a single frame for enrollment and returns UI status."""
        h_proc, w_proc = frame.shape[:2]
        faces = self.detector.detect_faces(frame)
        current_zone = None
        
        if faces:
            faces.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
            box = faces[0]
            norm_x = ((box[0] + box[2]) / 2) / w_proc
            norm_y = ((box[1] + box[3]) / 2) / h_proc
            current_zone = self.get_coverage_zone(norm_x, norm_y)
            
            # Enforce 10-frame interval between samples per specification
            if current_zone and self.coverage[current_zone] < self.target_per_zone and self.frame_count % 10 == 0:
                face_img = frame[box[1]:box[3], box[0]:box[2]]
                if face_img.size > 0:
                    spatial_emb = self.recognizer.get_spatial_embedding(face_img)
                    vlogger_emb = self.recognizer.get_vlogger_embedding(face_img)
                    
                    # Ensure Unit Normalization (norm=1) as specified
                    spatial_emb = spatial_emb / (np.linalg.norm(spatial_emb) + 1e-6)
                    vlogger_emb = vlogger_emb / (np.linalg.norm(vlogger_emb) + 1e-6)
                    
                    self.embeddings.append({"spatial": spatial_emb, "vlogger": vlogger_emb})
                    self.coverage[current_zone] += 1
                    print(f"[ENROLL] Captured sample for Zone {current_zone} ({self.coverage[current_zone]}/{self.target_per_zone})")

        self.frame_count += 1
        
        # Determine guidance
        next_target_zone = None
        for z in self.zones:
            if self.coverage[z] < self.target_per_zone:
                next_target_zone = z
                break
        
        is_complete = all(v >= self.target_per_zone for v in self.coverage.values())
        if is_complete:
            self.save()

        # Handle instruction persistence feedback
        instruction = "COMPLETE"
        if not is_complete:
            raw_instr = self.instr_map.get(next_target_zone, "Keep Still")
            # If user is in a zone that's ALREADY full, tell them to move.
            if current_zone and self.coverage[current_zone] >= self.target_per_zone and current_zone != "C":
                instruction = f"DONE! {raw_instr}"
            else:
                instruction = raw_instr

        return {
            "current_zone": current_zone,
            "target_zone": next_target_zone,
            "next_instruction": instruction,
            "coverage": self.coverage,
            "progress": sum(self.coverage.values()) / (len(self.zones) * self.target_per_zone),
            "is_complete": is_complete,
            "norm_x": norm_x * 100 if current_zone else None,
            "norm_y": norm_y * 100 if current_zone else None
        }

    def save(self):
        if len(self.embeddings) > 0:
            os.makedirs("data", exist_ok=True)
            np.save(self.save_path, np.array(self.embeddings, dtype=object))
            return True
        return False
