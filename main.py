import cv2
import numpy as np
import os
import time
import threading
import datetime
import queue
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import subprocess
import imageio_ffmpeg
from typing import Optional, List
from gui import VideoStream, VloggerGuardGUI
from detector import FaceDetector
from recognizer import FaceRecognizer
import csv
from enrollment import EnrollmentModule

class PerformanceLogger:
    """Logs vlogger identification and stability metrics to CSV."""
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_path = os.path.join(self.log_dir, "performance_log.csv")
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "total_faces", "vloggers_verified", "avg_trust_score", "privacy_enabled"])

    def log(self, total_faces, vloggers_verified, trust_score, privacy_enabled):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, total_faces, vloggers_verified, trust_score, int(privacy_enabled)])

class AudioRecorder(threading.Thread):
    """Background thread for synchronized audio recording."""
    def __init__(self, filename, samplerate=44100):
        super().__init__(daemon=True)
        self.filename = filename
        self.samplerate = samplerate
        self.recording = []
        self.stopped = False

    def callback(self, indata, frames, time, status):
        if not self.stopped:
            self.recording.append(indata.copy())

    def run(self):
        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.callback):
            while not self.stopped:
                sd.sleep(100)
        
        if self.recording:
            full_audio = np.concatenate(self.recording, axis=0)
            wav_write(self.filename, self.samplerate, full_audio)

    def stop(self):
        self.stopped = True
        self.join()

class VideoRecorder(threading.Thread):
    """Asynchronous video recorder to prevent UI/Capture blocking."""
    def __init__(self, filename, fps, frame_size):
        super().__init__(daemon=True)
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.frame_queue = queue.Queue(maxsize=300) 
        self.stopped = False
        self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.frame_size)

    def write(self, frame):
        if not self.stopped:
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass 

    def run(self):
        while not self.stopped or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
        self.writer.release()

    def stop(self):
        self.stopped = True
        self.join()

class InferenceResult:
    """Thread-safe container for bundled frame and detection results."""
    def __init__(self):
        self.vlogger_indices = []
        self.vlogger_boxes = [] # Stores confirmed [x1, y1, x2, y2]
        self.vlogger_buffer = 0
        self.lock = threading.Lock()
        # Stores (frame, faces) tuples
        self.sync_queue = queue.Queue(maxsize=1)
        
    def push(self, frame, faces):
        """Push bundled frame and faces, dropping older bundles to stay real-time."""
        try:
            while not self.sync_queue.empty():
                try: self.sync_queue.get_nowait()
                except queue.Empty: break
            self.sync_queue.put_nowait((frame.copy(), faces))
        except queue.Full: pass

    def get_bundle(self, timeout=0.01):
        """Get the latest bundle for processing."""
        try: return self.sync_queue.get(timeout=timeout)
        except queue.Empty: return None

    def update_recognition(self, vlogger_indices, vlogger_buffer, vlogger_boxes=None):
        """Update recognition output for the GUI."""
        with self.lock:
            self.vlogger_indices = vlogger_indices
            self.vlogger_buffer = vlogger_buffer
            if vlogger_boxes is not None:
                self.vlogger_boxes = vlogger_boxes

    def get_recognition(self):
        """Get the latest recognition results."""
        with self.lock:
            return self.vlogger_indices, self.vlogger_buffer, self.vlogger_boxes

class VloggerGuardApp:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.gui = VloggerGuardGUI()
        # Initial guess 1080p, VideoStream now handles set() before thread
        print("[INIT] Starting VideoStream at 1080p...")
        self.vs = VideoStream(width=1920, height=1080).start()
        
        self.profiles_path = "data/vlogger_profiles.npy"  # Changed to profiles (plural)
        self.vlogger_galleries = []  # List of profile galleries (max 2)
        self.privacy_enabled = True
        self.state = "STARTUP" 
        
        # Async Inference
        self.result: InferenceResult = InferenceResult()
        self.stop_inference: bool = False
        self.inference_thread: Optional[threading.Thread] = None
        self.buffer_max = 30 
        
        # Capture and Quality State
        self.capture_mode = "VIDEO" # "PHOTO" or "VIDEO"
        self.target_quality = "720p" # "320p", "720p", "1080p"
        self.quality_menu_open = False
        self._last_res = (0, 0)
        self.stop_inference = False
        self.last_log_time = time.time()
        
        # Recording State
        self.v_recorder: Optional[VideoRecorder] = None
        self.a_recorder: Optional[AudioRecorder] = None
        self.temp_v = None
        self.temp_a = None
        self.final_name = None
        self.recordings_dir = "recordings"
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        
        # Regions initialized dynamically in run() or update_layout()
        self.regions = {}
        
        # Performance Logger
        self.perf_logger = PerformanceLogger()
        self.last_log_time = time.time()
        self.video_rect = (0, 0, 1280, 720) # (ox, oy, w, h)

    def update_layout(self, win_w: int, win_h: int):
        """Calculates and updates UI regions based on window boundaries."""
        # Consistency scale based on a 720p height reference
        u_scale = win_h / 720.0
        
        icon_sz = int(45 * u_scale)
        btn_w, btn_h = int(100 * u_scale), int(35 * u_scale)
        rec_sz = int(60 * u_scale)
        slider_w, slider_h = int(115 * u_scale), int(30 * u_scale)
        
        # Landscape layout (Standard)
        shield_w = int(110 * u_scale)
        self.regions.update({
            "profile_icon": (10, 10, icon_sz, icon_sz),
            "folder_icon": (10 + icon_sz + 10, 10, icon_sz, icon_sz),
            "privacy_toggle": (win_w - shield_w - 10, 10, shield_w, btn_h),
            "quality_btn": (20, win_h - btn_h - 20, btn_w, btn_h),
            "record_btn": (win_w // 2 - rec_sz // 2, win_h - rec_sz - 15, rec_sz, rec_sz),
            "source_slider": (win_w - slider_w - 20, win_h - slider_h - 23, slider_w, slider_h),
            "icon_sz": icon_sz
        })
        # print(f"[LAYOUT] win_w:{win_w}, win_h:{win_h} | record_btn: {self.regions['record_btn']}")
        mx, my = 10, 10 + icon_sz + 10
        
        # Menu regions
        self.regions.update({
            "menu_profile1_change": (mx, my, int(90*u_scale), int(35*u_scale)),
            "menu_profile1_remove": (mx + int(95*u_scale), my, int(90*u_scale), int(35*u_scale)),
            "menu_profile2_change": (mx, my + int(45*u_scale), int(90*u_scale), int(35*u_scale)),
            "menu_profile2_remove": (mx + int(95*u_scale), my + int(45*u_scale), int(90*u_scale), int(35*u_scale)),
            "menu_add": (mx, my + int(85*u_scale), int(185*u_scale), int(35*u_scale)),
            "menu_q_320": (self.regions["quality_btn"][0], self.regions["quality_btn"][1] - int(105*u_scale), btn_w, int(30*u_scale)),
            "menu_q_720": (self.regions["quality_btn"][0], self.regions["quality_btn"][1] - int(70*u_scale), btn_w, int(30*u_scale)),
            "menu_q_1080": (self.regions["quality_btn"][0], self.regions["quality_btn"][1] - int(35*u_scale), btn_w, int(30*u_scale))
        })

    def load_profiles(self):
        """Load all enrolled profiles."""
        if os.path.exists(self.profiles_path):
            self.vlogger_galleries = list(np.load(self.profiles_path, allow_pickle=True))
            return len(self.vlogger_galleries) > 0
        return False

    def save_profiles(self):
        """Save all profiles to disk."""
        if not os.path.exists("data"):
            os.makedirs("data")
        np.save(self.profiles_path, np.array(self.vlogger_galleries, dtype=object))

    def add_profile(self):
        """Enroll a new profile (max 2) with retry limit."""
        if len(self.vlogger_galleries) >= 2:
            print("Maximum 2 profiles reached!")
            return False
        
        max_attempts = 3
        for attempt in range(max_attempts):
            cv2.destroyAllWindows()
            enroll = EnrollmentModule(self.detector, self.recognizer, self.vs, self.gui)
            if enroll.start_enrollment():
                # Load the newly created single profile and add to galleries
                if os.path.exists("data/vlogger_profile.npy"):
                    new_profile = np.load("data/vlogger_profile.npy")
                    self.vlogger_galleries.append(new_profile)
                    self.save_profiles()
                    print(f"Profile {len(self.vlogger_galleries)} added!")
                    return True
            else:
                print(f"Enrollment attempt {attempt + 1}/{max_attempts} failed.")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                else:
                    print("Enrollment failed after maximum attempts. Exiting.")
                    return False
        return False

    def delete_all_profiles(self):
        """Delete all profiles."""
        self.vlogger_galleries = []
        if os.path.exists(self.profiles_path):
            os.remove(self.profiles_path)

    def remove_profile(self, index):
        """Remove a specific profile by index."""
        if 0 <= index < len(self.vlogger_galleries):
            self.vlogger_galleries.pop(index)
            self.save_profiles()
            print(f"Profile {index + 1} removed.")

    def calculate_iou(self, boxA: tuple, boxB: tuple) -> float:
        """Standard Intersection over Union (IoU) calculation."""
        xA, yA, xB, yB = max(boxA[0],boxB[0]), max(boxA[1],boxB[1]), min(boxA[2],boxB[2]), min(boxA[3],boxB[3])
        interWidth, interHeight = max(0, xB - xA), max(0, yB - yA)
        if interWidth == 0 or interHeight == 0: return 0.0
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return (interWidth * interHeight) / float(areaA + areaB - (interWidth * interHeight))

    def inference_loop(self):
        """Worker thread with per-face persistence scoring (Smoothing)."""
        # List of {"box": (x1,y1,x2,y2), "score": int, "id": int}
        tracked_vloggers = [] 
        next_id = 0
        
        while not self.stop_inference:
            galleries = self.vlogger_galleries
            if self.state not in ["VLOGGING", "MENU"] or len(galleries) == 0:
                tracked_vloggers.clear()
                time.sleep(0.05)
                continue
            
            # FAST RECOVERY: Zero timeout if no vlogger is tracked
            timeout = 0.001 if (not tracked_vloggers) else 0.05
            bundle = self.result.get_bundle(timeout=timeout)
            if bundle is None: continue
            
            frame, faces = bundle
            if not faces:
                for tv in tracked_vloggers: tv["score"] = max(0, tv["score"] - 2)
                tracked_vloggers = [tv for tv in tracked_vloggers if tv["score"] > 0]
                self.result.update_recognition([], 0)
                continue

            vlogger_indices = []
            matched_faces = set()
            
            # 1. Match current faces to existing tracked vloggers (IoU)
            for face_idx, face_box in enumerate(faces):
                # 1. Update existing trackers using IoU (Maintenance Mode)
                best_iou = 0
                best_tv = None
                for tv in tracked_vloggers:
                    iou = self.calculate_iou(tv["box"], face_box)
                    # Relaxed maintenance (0.95) once already tracked
                    if iou > 0.4 and iou > best_iou:
                        best_iou = iou
                        best_tv = tv

                if best_tv:
                    matched_faces.add(face_idx)
                    best_tv["box"] = face_box
                    
                    # Periodic verification for stability if trust is low
                    if best_tv["score"] < self.buffer_max // 2:
                        x1, y1, x2, y2 = face_box
                        face_roi = frame[y1:y2, x1:x2]
                        if face_roi.size > 0:
                            emb = self.recognizer.get_embedding(face_roi)
                            # Check against all vlogger profiles (Corrected list iteration)
                            is_v = False
                            min_d = 1.0
                            for profile_embs in galleries:
                                match, d = self.recognizer.is_vlogger(emb, profile_embs, threshold=0.95)
                                if match:
                                    is_v = True
                                    min_d = min(min_d, d)
                            
                            if is_v: best_tv["score"] = min(self.buffer_max, best_tv["score"] + 8)
                            else: best_tv["score"] = max(0, best_tv["score"] - 4)
                    else:
                        # Natural "stickiness" while moving
                        best_tv["score"] = min(self.buffer_max, best_tv["score"] + 1)
            
            # 2. Check unmatched faces for NEW vloggers (Aggressive Search)
            unmatched_face_data = []
            for face_idx, face_box in enumerate(faces):
                if face_idx not in matched_faces:
                    area = (face_box[2]-face_box[0]) * (face_box[3]-face_box[1])
                    unmatched_face_data.append((face_idx, face_box, area))
            
            unmatched_face_data.sort(key=lambda x: x[2], reverse=True)

            # If no vlogger found yet, check more candidates (up to 4) to "scan fast"
            scan_limit = 4 if not tracked_vloggers else 2
            for face_idx, face_box, _ in unmatched_face_data[:scan_limit]:
                x1, y1, x2, y2 = face_box
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0: continue
                
                emb = self.recognizer.get_embedding(face_roi)
                # Acquisition Mode: Using lenient 0.85 threshold to catch side profiles
                vlogger_match = False
                found_dist = 1.0
                for profile_embs in galleries:
                    match, d = self.recognizer.is_vlogger(emb, profile_embs, threshold=0.85)
                    if match:
                        vlogger_match = True
                        found_dist = min(found_dist, d)
                
                if vlogger_match:
                    # Score based on match quality: 0.85 threshold -> higher dist = lower initial trust
                    initial_score = self.buffer_max if found_dist < 0.7 else self.buffer_max // 2
                    tracked_vloggers.append({"box": face_box, "score": initial_score, "id": next_id})
                    next_id += 1
                    matched_faces.add(face_idx)
                    # If we found the vlogger, stop scanning others this frame to save CPU
                    break 

            # 3. Decay unmatched tracked vloggers (Extended memory: 30 frame buffer)
            active_ids = {tv["id"] for i, tv in enumerate(tracked_vloggers) if any(self.calculate_iou(tv["box"], f) > 0.3 for f in faces)}
            for tv in tracked_vloggers:
                if tv["id"] not in active_ids:
                    tv["score"] = max(0, tv["score"] - 1) # Slow decay (-1) for 30-frame persistence

            # 4. Filter and prepare vlogger data for GUI
            tracked_vloggers = [tv for tv in tracked_vloggers if tv["score"] > 0]
            
            final_vlogger_indices = []
            final_vlogger_boxes = [] # For Shadow Box Tracking
            max_score = 0
            for face_idx, face_box in enumerate(faces):
                for tv in tracked_vloggers:
                    if self.calculate_iou(tv["box"], face_box) > 0.4:
                        final_vlogger_indices.append(face_idx)
                        final_vlogger_boxes.append(face_box)
                        max_score = max(max_score, tv["score"])
                        break
            
            self.result.update_recognition(final_vlogger_indices, max_score, final_vlogger_boxes)

    def draw_hud(self, frame, face_count, fps):
        h, w = frame.shape[:2]
        u_scale = h / 720.0
        
        # Ribbons at absolute edges
        ribbon_top_h = int(55 * u_scale)
        ribbon_bot_h = int(100 * u_scale)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, ribbon_top_h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (0, h - ribbon_bot_h), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Profile/Folder Icons
        px, py, pw, ph = self.regions["profile_icon"]
        cv2.circle(frame, (px + pw//2, py + ph//2), int(18*u_scale), (200, 200, 200), -1)
        cv2.putText(frame, str(len(self.vlogger_galleries)), (px + pw - 8, py + ph - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4*u_scale, (0, 255, 255), max(1, int(u_scale)))
        
        fx, fy, fw, fh = self.regions["folder_icon"]
        # Simplified icon
        cv2.rectangle(frame, (fx+int(5*u_scale), fy+int(10*u_scale)), (fx+int(35*u_scale), fy+int(35*u_scale)), (200, 200, 100), max(1, int(2*u_scale)))
        
        tx, ty, tw, th = self.regions["privacy_toggle"]
        btn_color = (0, 180, 0) if self.privacy_enabled else (0, 0, 180)
        cv2.rectangle(frame, (tx, ty), (tx+tw, ty+th), btn_color, -1, cv2.LINE_AA)
        status_text = "SHIELD: ON" if self.privacy_enabled else "SHIELD: OFF"
        cv2.putText(frame, status_text, (tx+int(10*u_scale), ty+int(23*u_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*u_scale, (255, 255, 255), max(1, int(u_scale)))
        
        hud_txt = f"FPS: {int(fps)} | FACES: {face_count}"
        txt_x, txt_y = int(120*u_scale), int(32*u_scale)
        cv2.putText(frame, hud_txt, (txt_x, txt_y), cv2.FONT_HERSHEY_DUPLEX, 0.5*u_scale, (0, 255, 255), max(1, int(u_scale)))
        
        # Capture Mode Slider
        sx, sy, sw, sh = self.regions["source_slider"]
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (60, 60, 60), -1, cv2.LINE_AA)
        thumb_x = sx if self.capture_mode == "PHOTO" else sx + sw//2
        cv2.rectangle(frame, (thumb_x, sy), (thumb_x + sw//2, sy+sh), (100, 100, 255), -1)
        cv2.putText(frame, "PIC", (sx+5, sy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "VID", (sx+sw//2+5, sy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Quality Button
        qx, qy, qw, qh = self.regions["quality_btn"]
        cv2.rectangle(frame, (qx, qy), (qx+qw, qy+qh), (80, 80, 80), -1, cv2.LINE_AA)
        cv2.putText(frame, self.target_quality, (qx+15, qy+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_menu(self, frame):
        h, w = frame.shape[:2]
        u_scale = h / 720.0
        overlay = frame.copy()
        
        if self.state == "MENU":
            mx, my, mw, mh = self.regions["menu_profile1_change"]
            menu_w = int(210 * u_scale)
            menu_h = int((50 + (len(self.vlogger_galleries) * 45) + (50 if len(self.vlogger_galleries) < 2 else 10)) * u_scale)
            cv2.rectangle(overlay, (mx - int(10*u_scale), my - int(10*u_scale)), (mx + menu_w, my + menu_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            y_offset = my
            for i in range(len(self.vlogger_galleries)):
                cv2.putText(frame, f"Profile {i+1}", (mx + int(5*u_scale), y_offset - int(5*u_scale)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45*u_scale, (200, 200, 200), max(1, int(u_scale)))
                cx, cy, cw, ch = self.regions[f"menu_profile{i+1}_change"]
                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (60, 100, 60), -1)
                cv2.putText(frame, "CHANGE", (cx+int(15*u_scale), cy+int(22*u_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.4*u_scale, (255, 255, 255), max(1, int(u_scale)))
                rx, ry, rw, rh = self.regions[f"menu_profile{i+1}_remove"]
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (100, 40, 40), -1)
                cv2.putText(frame, "REMOVE", (rx+int(15*u_scale), ry+int(22*u_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.4*u_scale, (255, 255, 255), max(1, int(u_scale)))
                y_offset += int(45 * u_scale)
            if len(self.vlogger_galleries) < 2:
                ax, ay, aw, ah = self.regions["menu_add"]
                cv2.rectangle(frame, (ax, ay), (ax+aw, ay+ah), (40, 80, 120), -1)
                cv2.putText(frame, "+ ADD PROFILE", (ax+int(30*u_scale), ay+int(23*u_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*u_scale, (255, 255, 255), max(1, int(u_scale)))

        if self.quality_menu_open:
            qx, qy, qw, qh = self.regions["menu_q_1080"]
            cv2.rectangle(overlay, (qx - int(5*u_scale), qy - int(5*u_scale)), (qx + qw + int(5*u_scale), qy + int(110*u_scale)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            for q in ["1080", "720", "320"]:
                reg = self.regions[f"menu_q_{q}"]
                rx, ry, rw, rh = reg
                color = (100, 100, 255) if self.target_quality == f"{q}p" else (60, 60, 60)
                cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, -1)
                cv2.putText(frame, f"{q}p", (rx+int(25*u_scale), ry+int(20*u_scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.45*u_scale, (255, 255, 255), max(1, int(u_scale)))
        
        return frame

    def draw_record_button(self, frame):
        h = frame.shape[0]
        u_scale = h / 720.0
        rx, ry, rw, rh = self.regions["record_btn"]
        active = self.v_recorder is not None
        color = (0, 0, 200) if active else (150, 150, 150)
        
        if active and int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (rx-int(2*u_scale), ry-int(2*u_scale)), (rx+rw+int(2*u_scale), ry+rh+int(2*u_scale)), (0, 0, 255), max(1, int(2*u_scale)))
        
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (40, 40, 40), -1)
        cv2.circle(frame, (rx + rw//2, ry + rh//2), int(12*u_scale), color, -1)
        
        if self.capture_mode == "PHOTO":
            txt = "SNAP"
        else:
            txt = "STOP" if active else "REC"
            
        cv2.putText(frame, txt, (rx + int(45*u_scale), ry + rh//2 + int(7*u_scale)), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6*u_scale, (255, 255, 255), max(1, int(u_scale)))
        return frame

    def mux_video_audio(self, video_path, audio_path, output_path):
        """Merges video and audio using imageio-ffmpeg."""
        print(f"Merging flows into {output_path}...")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(video_path)
            os.remove(audio_path)
            print("Video finalized with audio.")
        except Exception as e:
            print(f"Merge failed: {e}")

    def toggle_recording(self, frame_size):
        if self.v_recorder is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_v = os.path.join(self.recordings_dir, f"temp_{ts}.avi")
            temp_a = os.path.join(self.recordings_dir, f"temp_{ts}.wav")
            self.final_name = os.path.join(self.recordings_dir, f"vlog_{ts}.mp4")
            
            self.v_recorder = VideoRecorder(temp_v, 30.0, frame_size)
            self.v_recorder.start()
            self.a_recorder = AudioRecorder(temp_a)
            self.a_recorder.start()
            
            self.temp_v = temp_v
            self.temp_a = temp_a
            print("Synchronized A/V recording started.")
        else:
            v_path, a_path, out_path = self.temp_v, self.temp_a, self.final_name
            self.v_recorder.stop()
            self.a_recorder.stop()
            self.v_recorder = None
            self.a_recorder = None
            threading.Thread(target=self.mux_video_audio, args=(v_path, a_path, out_path), daemon=True).start()

    def capture_photo(self, frame):
        """Saves the current (blurred) frame as a JPEG."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recordings_dir, f"snap_{ts}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Photo saved: {filename}")

    def handle_clicks(self, click, fps, frame):
        if not click: return
        x, y = click
        print(f"[CLICK] Window X:{x}, Y:{y} | State: {self.state} | QualMenu: {self.quality_menu_open}")

        if self.quality_menu_open:
            for q in ["320", "720", "1080"]:
                rx, ry, rw, rh = self.regions[f"menu_q_{q}"]
                if rx < x < rx+rw and ry < y < ry+rh:
                    self.target_quality = f"{q}p"
                    self.quality_menu_open = False
                    print(f"[UI] Quality changed to {self.target_quality}")
                    return
            # Close if click outside
            qx, qy, qw, qh = self.regions["menu_q_320"]
            # Menu area spans from menu_q_1080 to menu_q_320
            q1080 = self.regions["menu_q_1080"]
            if not (q1080[0] - 5 < x < q1080[0] + q1080[2] + 5 and q1080[1] - 5 < y < qy + rh + 5):
                print("[UI] Closing Quality Menu (clicked outside)")
                self.quality_menu_open = False
                # Continue checking other buttons instead of returning

        if self.state == "VLOGGING":
            px, py, pw, ph = self.regions["profile_icon"]
            if px < x < px+pw and py < y < py+ph:
                print("[UI] Profile Icon Clicked -> MENU")
                self.state = "MENU"
                return
            
            fx, fy, fw, fh = self.regions["folder_icon"]
            if fx < x < fx+fw and fy < y < fy+fh:
                print("[UI] Folder Icon Clicked")
                try: os.startfile(os.path.abspath(self.recordings_dir))
                except: pass
                return
            
            tx, ty, tw, th = self.regions["privacy_toggle"]
            if tx < x < tx+tw and ty < y < ty+th:
                self.privacy_enabled = not self.privacy_enabled
                print(f"[UI] Privacy Toggle: {self.privacy_enabled}")
                return
            
            rx, ry, rw, rh = self.regions["record_btn"]
            if rx < x < rx+rw and ry < y < ry+rh:
                print("[UI] Record/Snap Clicked")
                if self.capture_mode == "PHOTO":
                    ox, oy, vw, vh = self.video_rect
                    # Safety check for crop
                    if vh > 0 and vw > 0:
                        cap_frame = frame[oy:oy+vh, ox:ox+vw] if frame is not None else None
                    else:
                        cap_frame = frame
                    self.capture_photo(cap_frame)
                else:
                    self.toggle_recording((frame.shape[1], frame.shape[0]))
                return
            
            sx, sy, sw, sh = self.regions["source_slider"]
            if sx < x < sx+sw and sy < y < sy+sh:
                old_mode = self.capture_mode
                self.capture_mode = "PHOTO" if x < sx + sw//2 else "VIDEO"
                if old_mode != self.capture_mode:
                    print(f"[UI] Capture mode changed to {self.capture_mode}")
                return

            qx, qy, qw, qh = self.regions["quality_btn"]
            if qx < x < qx+qw and qy < y < qy+qh:
                self.quality_menu_open = not self.quality_menu_open
                print(f"[UI] Quality Menu Toggle: {self.quality_menu_open}")
                return
            # Close if click outside
            qx, qy, qw, qh = self.regions["menu_q_320"]
            if not (qx - 5 < x < qx + qw + 5 and qy - 5 < y < qy + 110):
                self.quality_menu_open = False

        if self.state == "MENU":
            # Check profile-specific buttons
            for i in range(len(self.vlogger_galleries)):
                # Change button
                cx, cy, cw, ch = self.regions[f"menu_profile{i+1}_change"]
                if cx < x < cx+cw and cy < y < cy+ch:
                    # Remove old profile and enroll new one
                    self.remove_profile(i)
                    if self.add_profile():
                        self.state = "VLOGGING"
                    cv2.namedWindow(self.gui.window_name)
                    cv2.setMouseCallback(self.gui.window_name, self.gui._mouse_callback)
                    return
                
                # Remove button
                rx, ry, rw, rh = self.regions[f"menu_profile{i+1}_remove"]
                if rx < x < rx+rw and ry < y < ry+rh:
                    self.remove_profile(i)
                    if len(self.vlogger_galleries) == 0:
                        self.state = "ENROLLING"
                    return
            
            # Add profile button
            if len(self.vlogger_galleries) < 2:
                ax, ay, aw, ah = self.regions["menu_add"]
                if ax < x < ax+aw and ay < y < ay+ah:
                    if self.add_profile():
                        self.state = "VLOGGING"
                    cv2.namedWindow(self.gui.window_name)
                    cv2.setMouseCallback(self.gui.window_name, self.gui._mouse_callback)
                    return
            
            # Click outside menu to close
            menu_height = 50 + (len(self.vlogger_galleries) * 45) + (50 if len(self.vlogger_galleries) < 2 else 10)
            if x > 210 or y > menu_height or y < 50:
                self.state = "VLOGGING"

    def run(self):
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.inference_thread.start()
        
        self._last_res = (0, 0)
        self._vs_is_cam = True
        
        while True:
            # 1. Get current window size and update display regions first!
            win_w, win_h = self.gui.get_window_size()
            self.update_layout(win_w, win_h)
            
            # 2. Read frame
            ret, frame_raw = self.vs.read()
            if not ret or frame_raw is None:
                continue
            
            # 3. Handle interactions using updated window-relative hitboxes
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Interaction uses raw frame size for photo crops later
            self.handle_clicks(self.gui.get_click(), fps, frame_raw)
            
            # 4. Processing flow
            frame = cv2.flip(frame_raw, 1)
            
            # Resize to Target Quality for Processing (Inference/Blur)
            h_orig, w_orig = frame.shape[:2]
            target_h = int(self.target_quality.replace("p", ""))
            
            if h_orig != target_h:
                aspect = w_orig / h_orig
                target_w = int(target_h * aspect)
                interp = cv2.INTER_AREA if target_h < h_orig else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (target_w, target_h), interpolation=interp)
            
            h_proc, w_proc = frame.shape[:2]
            
            if self.state == "STARTUP":
                if self.load_profiles(): self.state = "VLOGGING"
                else: self.state = "ENROLLING"
            elif self.state == "ENROLLING":
                if self.add_profile():
                    self.state = "VLOGGING"
                else:
                    self.state = "VLOGGING" if len(self.vlogger_galleries) > 0 else "STARTUP"
                cv2.namedWindow(self.gui.window_name)
                cv2.setMouseCallback(self.gui.window_name, self.gui._mouse_callback)
            elif self.state in ["VLOGGING", "MENU"]:
                # INSTANT DETECTION in main thread (no lag!)
                faces = self.detector.detect_faces(frame)
                
                # Bundled Frame/Face push for perfect sync and zero-lag recovery
                self.result.push(frame, faces)
                
                # Get recognition results from background thread (including confirmed boxes)
                v_indices, v_buf, v_boxes = self.result.get_recognition()
                
                # Shadow Box Tracking: Instantly unblur if face matches a confirmed position
                shadow_vlogger_indices = set(v_indices)
                for i, face_box in enumerate(faces):
                    if i in shadow_vlogger_indices: continue
                    for confirmed_box in v_boxes:
                        if self.calculate_iou(face_box, confirmed_box) > 0.7:
                            shadow_vlogger_indices.add(i)
                            break

                if self.privacy_enabled:
                    for i, (x1, y1, x2, y2) in enumerate(faces):
                        if i not in shadow_vlogger_indices:
                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0: frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
                
                if self.v_recorder is not None:
                    if fps_counter % int(max(1, fps/30.0)) == 0: self.v_recorder.write(frame)
                
                # 5. Create Display Canvas matching the window
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                
                # 2. Scale and center the processed video frame (maintain 16:9)
                h_proc, w_proc = frame.shape[:2]
                # The video itself will be 16:9
                v_aspect = 16/9
                # Calculate maximum size that fits in window while keeping 16:9
                if win_w / win_h > v_aspect:
                    # Window is wider than 16:9: height is limiting
                    v_h = win_h
                    v_w = int(v_h * v_aspect)
                else:
                    # Window is taller than 16:9: width is limiting
                    v_w = win_w
                    v_h = int(v_w / v_aspect)
                
                # Center positions
                ox = (win_w - v_w) // 2
                oy = (win_h - v_h) // 2
                self.video_rect = (ox, oy, v_w, v_h)
                
                # Resize video frame to fit the calculated area
                resized_video = cv2.resize(frame, (v_w, v_h), interpolation=cv2.INTER_LINEAR)
                # Place video on canvas
                canvas[oy:oy+v_h, ox:ox+v_w] = resized_video
                
                # 4. Draw Face Labels relative to the CENTERED video
                scale_x = v_w / w_proc
                scale_y = v_h / h_proc
                
                if self.privacy_enabled:
                    for i, (x1, y1, x2, y2) in enumerate(faces):
                        # Map processed coords to window-absolute centered video coords
                        dx1, dy1 = int(ox + x1 * scale_x), int(oy + y1 * scale_y)
                        dx2, dy2 = int(ox + x2 * scale_x), int(oy + y2 * scale_y)
                        
                        if i not in shadow_vlogger_indices:
                            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 0, 255), 1)
                            cv2.putText(canvas, "ANONYMIZED", (dx1, dy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        else:
                            color, l, t = (0, 255, 0), 20, 2
                            cv2.line(canvas, (dx1,dy1), (dx1+l,dy1), color, t)
                            cv2.line(canvas, (dx1,dy1), (dx1,dy1+l), color, t)
                            cv2.line(canvas, (dx2,dy1), (dx2-l,dy1), color, t)
                            cv2.line(canvas, (dx2,dy1), (dx2,dy1+l), color, t)
                            cv2.line(canvas, (dx1,dy2), (dx1+l,dy2), color, t)
                            cv2.line(canvas, (dx1,dy2), (dx1,dy2-l), color, t)
                            cv2.line(canvas, (dx2,dy2), (dx2-l,dy2), color, t)
                            cv2.line(canvas, (dx2,dy2), (dx2,dy2-l), color, t)
                            
                            is_verified = i in v_indices
                            lbl = "VLOGGER VERIFIED" if is_verified else "INSTANT RECOGNITION"
                            cv2.putText(canvas, lbl, (dx1, dy1-15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                
                # 5. Draw HUD at the window edges
                self.draw_hud(canvas, len(faces), fps)
                self.draw_record_button(canvas)
                if self.state == "MENU" or self.quality_menu_open: 
                    self.draw_menu(canvas)
                
                self.gui.show_frame(canvas)
                
                # Periodic Logging (Every 1 second)
                if time.time() - self.last_log_time > 1.0:
                    v_verified = len(v_indices)
                    self.perf_logger.log(len(faces), v_verified, v_buf, self.privacy_enabled)
                    self.last_log_time = time.time()
            
            if self.gui.check_exit(): break
        
        self.stop_inference = True
        if self.v_recorder is not None: 
            self.v_recorder.stop()
        if self.a_recorder is not None: 
            self.a_recorder.stop()
        if self.vs is not None:
            self.vs.release()
        self.gui.close()

if __name__ == "__main__":
    VloggerGuardApp().run()
