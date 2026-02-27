import cv2
import numpy as np
import os
import time
from detector import FaceDetector
from recognizer import FaceRecognizer

class EnrollmentModule:
    def __init__(self, detector=None, recognizer=None, cap=None, gui=None):
        self.detector = detector if detector else FaceDetector()
        self.recognizer = recognizer if recognizer else FaceRecognizer()
        self.cap = cap
        self.gui = gui
        self.save_path = "data/vlogger_profile.npy"
        
        # Video sweep parameters
        self.phase_duration = 6  # seconds per phase
        self.frame_sample_interval = 3  # sample every 3 frames
        self.min_embeddings = 10  # lowered to 10 (5 per phase)
        self.max_embeddings = 40

    def get_coverage_zone(self, norm_x: float, norm_y: float) -> str:
        """Map normalized face position to one of 9 coverage zones (Screen-Relative)."""
        # dx < 0 means face is on the Left side of the mirrored screen
        # dy < 0 means face is on the Top side of the screen
        dx = norm_x - 0.5
        dy = norm_y - 0.5
        
        # Consistent thresholds
        thresh = 0.035 
        diag_thresh = 0.025
        
        # Center check
        if abs(dx) < thresh and abs(dy) < thresh:
            return "C"
            
        # Determine Screen-Relative directions
        if dy < -thresh: # Screen Top
            if dx < -diag_thresh: return "NW" # Top-Left
            if dx > diag_thresh: return "NE"  # Top-Right
            return "N"
        elif dy > thresh: # Screen Bottom
            if dx < -diag_thresh: return "SW" # Bottom-Left
            if dx > diag_thresh: return "SE"  # Bottom-Right
            return "S"
        else: # Screen Middle (Horizontal Only)
            if dx < -thresh: return "W" # Left
            if dx > thresh: return "E"  # Right
            
        return "C"

    def start_enrollment(self):
        """Redesigned Hemisphere Coverage Enrollment (Screen-Relative)."""
        print("Starting Hemisphere Coverage Enrollment...")
        is_shared_vs = hasattr(self.cap, 'read') and not isinstance(self.cap, cv2.VideoCapture)
        cap = self.cap if self.cap else cv2.VideoCapture(0)
        
        # 9 Zones: Center, North, NE, East, SE, South, SW, West, NW
        zones = ["C", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        coverage = {z: 0 for z in zones}
        target_per_zone = 3
        embeddings = []
        frame_count = 0
        
        # Compass UI positions (Screen-Relative: West=Left, East=Right)
        compass_map = {
            "C":  (0, 0),
            "N":  (0, -1),
            "NE": (1, -1),  # Top-Right
            "E":  (1, 0),   # Middle-Right
            "SE": (1, 1),   # Bottom-Right
            "S":  (0, 1),
            "SW": (-1, 1),  # Bottom-Left
            "W":  (-1, 0),  # Middle-Left
            "NW": (-1, -1)  # Top-Left
        }
        
        # Redo Logic
        redo_clicked = [False]
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 10 <= x <= 90 and 10 <= y <= 45:
                    redo_clicked[0] = True
        
        cv2.namedWindow("Enrollment Feed")
        cv2.setMouseCallback("Enrollment Feed", on_mouse)

        while True:
            if is_shared_vs: success, frame = self.cap.read()
            else: success, frame = cap.read()
            if not success: break
            
            # Handle Redo Reset
            if redo_clicked[0]:
                embeddings = []
                coverage = {z: 0 for z in zones}
                redo_clicked[0] = False
                print("Enrollment Reset.")
            
            # Mirror feed
            frame = cv2.flip(frame, 1)
            h_proc, w_proc = frame.shape[:2]
            
            # 720p Display Scale for uniform UI
            disp_h = 720
            disp_w = int(disp_h * (w_proc / h_proc))
            display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            h, w = display_frame.shape[:2]
            
            faces = self.detector.detect_faces(frame)
            
            # 1. Darkened Overlay with Guideline
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (int(w*0.15), int(h*0.25)), 0, 0, 360, 255, -1)
            overlay = display_frame.copy()
            overlay[mask == 0] = (overlay[mask == 0] * 0.3).astype(np.uint8)
            display_frame = overlay
            cv2.ellipse(display_frame, (w//2, h//2), (int(w*0.15), int(h*0.25)), 0, 0, 360, (0, 255, 255), 2)

            # REDO Button (Top-Left)
            redo_rect = (10, 10, 80, 35)
            cv2.rectangle(display_frame, (redo_rect[0], redo_rect[1]), 
                         (redo_rect[0]+redo_rect[2], redo_rect[1]+redo_rect[3]), (50, 50, 150), -1)
            cv2.putText(display_frame, "REDO", (redo_rect[0]+15, redo_rect[1]+25), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # 2. Coverage Compass UI & Guidance Arrow
            compass_x, compass_y = w - 100, 100
            spacing = 25
            cv2.putText(display_frame, "COVERAGE", (compass_x-35, compass_y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Find closest incomplete zone to guide user
            next_zone = None
            for z in zones:
                if coverage[z] < target_per_zone:
                    next_zone = z
                    break
            
            # Draw guidance arrow toward next_zone dot
            if next_zone and next_zone != "C":
                pos = compass_map[next_zone]
                # Arrow points towards the screen-relative target dot
                start_pt = (w // 2, h // 2)
                end_pt = (w // 2 + pos[0] * 80, h // 2 + pos[1] * 80)
                
                # Pulsing arrow color
                color = (0, int(150 + 100 * np.sin(time.time() * 5)), 255)
                cv2.arrowedLine(display_frame, start_pt, end_pt, color, 3, tipLength=0.2)
                
                # PHYSICAL Instructions (Mirror-Aware)
                # To move face RIGHT on mirrored screen, look LEFT physically.
                instr_map = {
                    "N": "Look UP", "S": "Look DOWN", 
                    "E": "Look LEFT", "W": "Look RIGHT",
                    "NE": "Look UP-LEFT", "NW": "Look UP-RIGHT", 
                    "SE": "Look DOWN-LEFT", "SW": "Look DOWN-RIGHT"
                }
                cv2.putText(display_frame, instr_map.get(next_zone, ""), (w//2 - 70, h//2 + 100), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

            current_zone = None
            if faces:
                faces.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
                box = faces[0]
                norm_x = ((box[0] + box[2]) / 2) / w
                norm_y = ((box[1] + box[3]) / 2) / h
                current_zone = self.get_coverage_zone(norm_x, norm_y)
                
                # Feedback on current detected zone
                cv2.putText(display_frame, f"CURRENT: {current_zone}", (w//2 - 40, h//2 + 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Auto-capture if in a new/needing zone with interval for variance
                if coverage[current_zone] < target_per_zone and frame_count % 10 == 0:
                    face_img = frame[box[1]:box[3], box[0]:box[2]]
                    if face_img.size > 0:
                        emb = self.recognizer.get_embedding(face_img)
                        embeddings.append(emb)
                        coverage[current_zone] += 1

            # Draw Compass Dots
            for z, pos in compass_map.items():
                zx = compass_x + pos[0] * spacing
                zy = compass_y + pos[1] * spacing
                
                # Color based on status
                color = (100, 100, 100) # Empty
                if coverage[z] >= target_per_zone: color = (0, 255, 0) # Filled
                elif coverage[z] > 0: color = (0, 255, 255) # Partial
                
                # Highlight if active
                radius = 6 if z != current_zone else 10
                thick = -1 if coverage[z] > 0 else 1
                cv2.circle(display_frame, (zx, zy), radius, color, thick)
                if z == current_zone: cv2.circle(display_frame, (zx, zy), radius+2, (255, 255, 255), 1)

            # 3. Status Text
            total_samples = sum(coverage.values())
            progress = total_samples / (len(zones) * target_per_zone)
            cv2.putText(display_frame, f"SAMPLES: {total_samples}/{len(zones)*target_per_zone}", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "LOOK IN ALL DIRECTIONS UNTIL COMPASS TURNS GREEN", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Progress bar
            cv2.rectangle(display_frame, (40, h-30), (w-40, h-10), (60, 60, 60), -1)
            cv2.rectangle(display_frame, (40, h-30), (40 + int((w-80)*progress), h-10), (0, 255, 0), -1)

            cv2.imshow("Enrollment Feed", display_frame)
            frame_count += 1
            if all(v >= target_per_zone for v in coverage.values()): break
            if cv2.waitKey(1) & 0xFF == ord('q'): return False

        if not is_shared_vs: cap.release()
        cv2.destroyAllWindows()
        
        # Save and Feedback
        if len(embeddings) >= self.min_embeddings:
            np.save(self.save_path, np.array(embeddings))
            # Success feedback
            success_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            success_frame[:] = (0, 100, 0)
            cv2.putText(success_frame, "FULL COVERAGE SECURED!", (80, 220), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(success_frame, f"{len(embeddings)} multi-angle samples saved", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            cv2.imshow("Enrollment Feed", success_frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            return True
        return False
