import cv2
import numpy as np
import os
import time
import threading
import datetime
import queue
try:
    import sounddevice as sd
except ImportError:
    sd = None
import wave
import subprocess
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None
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
        if sd is None:
            print("[AUDIO] sounddevice not available. Skipping recording.")
            return
        try:
            with sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.callback):
                while not self.stopped:
                    sd.sleep(100)
        except Exception as e:
            print(f"[AUDIO] InputStream Error: {e}")
        
        # Ensure file is written even if empty/errored to satisfy FFmpeg
        try:
            if self.recording:
                full_audio = np.concatenate(self.recording, axis=0)
                with wave.open(self.filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(self.samplerate)
                    # Convert float32 to int16
                    audio_int16 = (full_audio * 32767).astype(np.int16)
                    wf.writeframes(audio_int16.tobytes())
            else:
                # Write 1 second of silence
                with wave.open(self.filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.samplerate)
                    wf.writeframes(np.zeros((self.samplerate, 1), dtype=np.int16).tobytes())
        except Exception as e:
            print(f"[AUDIO] Save Error: {e}")

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
        last_frame = None
        frame_interval = 1.0 / self.fps
        next_time = time.time()
        
        while not self.stopped or not self.frame_queue.empty():
            current_time = time.time()
            if current_time >= next_time:
                try:
                    # Get latest available frame or reuse last frame (CFR logic)
                    if not self.frame_queue.empty():
                        last_frame = self.frame_queue.get_nowait()
                        self.frame_queue.task_done()
                    
                    if last_frame is not None:
                        self.writer.write(last_frame)
                    
                    next_time += frame_interval
                except Exception as e:
                    print(f"[VIDEO] Write Error: {e}")
            else:
                # Short sleep to prevent CPU spin
                time.sleep(0.001)
                
        self.writer.release()

    def stop(self):
        self.stopped = True
        self.join()

class InferenceResult:
    """Thread-safe container for bundled frame and detection results."""
    def __init__(self):
        self.vlogger_map = {} # Maps slot_index to face_box
        self.vlogger_scores = {} # Maps slot_index to integrity_score
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

    def update_resolution(self, vlogger_map, vlogger_scores):
        """Update identity resolution output for the GUI."""
        with self.lock:
            self.vlogger_map = vlogger_map.copy()
            self.vlogger_scores = vlogger_scores.copy()

    def get_resolution(self):
        """Get the latest resolution results."""
        with self.lock:
            return self.vlogger_map, self.vlogger_scores

class VloggerGuardApp:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer() # Now using ArcFace V2
        self.gui = VloggerGuardGUI()
        print("[INIT] Starting VideoStream at 1080p...")
        self.vs = VideoStream(width=1920, height=1080).start()
        
        self.profiles_path = "data/vlogger_profiles.npy"
        self.vlogger_galleries = []  
        self.privacy_enabled = True
        self.state = "STARTUP" 
        
        # Identity Resolution State
        self.result: InferenceResult = InferenceResult()
        self.stop_inference: bool = False
        self.inference_thread: Optional[threading.Thread] = None
        self.integrity_max = 100
        self.integrity_threshold = 75 # Threshold to "unblur"
        
        # Tracking State (Persistent between loop iterations)
        self.track_vloggers = {} # Maps profile_index to {"box": box, "score": int, "track_id": int}
        
        # Capture and Quality State
        self.capture_mode = "VIDEO"
        self.target_quality = "720p"
        self.quality_menu_open = False
        self._last_res = (0, 0)
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
        """Enroll a new profile with matching detection for profile merging."""
        max_attempts = 2
        for attempt in range(max_attempts):
            cv2.destroyAllWindows()
            enroll = EnrollmentModule(self.detector, self.recognizer, self.vs, self.gui)
            if enroll.start_enrollment():
                # 1. Load the newly captured embeddings
                if os.path.exists("data/vlogger_profile.npy"):
                    new_embs = list(np.load("data/vlogger_profile.npy", allow_pickle=True))
                    
                    # 2. Check for matches in existing profiles
                    merged = False
                    # Representative embedding from new set (mean or first few)
                    sample_emb = new_embs[0]
                    
                    for i, existing_profile in enumerate(self.vlogger_galleries):
                        # Use MobileFaceNet (vlogger) for merge check
                        existing_vlogger_embs = [p["vlogger"] for p in existing_profile]
                        match, dist = self.recognizer.is_vlogger(sample_emb["vlogger"], existing_vlogger_embs, threshold=0.40)
                        if match:
                            # MERGE: Append new unique/diverse embeddings to existing profile
                            # For simplicity, we just extend and take a subset if too large
                            updated_profile = list(existing_profile) + new_embs
                            # Keep it robust but efficient (max 80 embeddings)
                            if len(updated_profile) > 80:
                                updated_profile = updated_profile[-80:]
                            
                            self.vlogger_galleries[i] = np.array(updated_profile, dtype=object)
                            self.save_profiles()
                            print(f"[IDENTITY] Profile {i+1} merged and strengthened (Dist: {dist:.3f})")
                            merged = True
                            break
                    
                    if not merged:
                        if len(self.vlogger_galleries) < 2:
                            self.vlogger_galleries.append(np.array(new_embs, dtype=object))
                            self.save_profiles()
                            print(f"[IDENTITY] New Vlogger Profile Added and Saved (Slot {len(self.vlogger_galleries)})")
                        else:
                            print("[WARNING] Max 2 unique vloggers reached. New identity discarded.")
                    
                    # Clean up temp file
                    try: os.remove("data/vlogger_profile.npy")
                    except: pass
                    print("[IDENTITY] Enrollment process complete.")
                    return True
            else:
                print(f"Enrollment attempt {attempt + 1}/{max_attempts} was canceled or failed.")
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
        """
        Global Identity Resolution Engine.
        Handles multi-vlogger consistency, spatial inertia, and strict privacy scoring.
        """
        track_states = {} # Stores last known state for each profile slot {idx: {"box": box, "score": int, "missed": int}}
        
        while not self.stop_inference:
            galleries = self.vlogger_galleries
            if self.state not in ["VLOGGING", "MENU"] or not galleries:
                track_states.clear()
                time.sleep(0.05)
                continue
            
            # Real-time bundle retrieval
            bundle = self.result.get_bundle(timeout=0.05)
            if bundle is None: continue
            
            frame, faces = bundle
            h_proc, w_proc = frame.shape[:2]
            
            # Current assignments for this frame
            current_assignments = {} # {face_idx: profile_idx}
            vlogger_map = {} # {profile_idx: box}
            vlogger_scores = {} # {profile_idx: score}

            if not faces:
                # Decay existing scores
                for idx in list(track_states.keys()):
                    track_states[idx]["score"] = max(0, track_states[idx]["score"] - 5)
                    track_states[idx]["missed"] += 1
                    if track_states[idx]["score"] == 0 and track_states[idx]["missed"] > 30:
                        del track_states[idx]
                self.result.update_resolution({}, {})
                continue

            # 1. Distance Matrix Construction (Faces vs Profiles)
            # matrix[face_idx][profile_idx] = distance
            dist_matrix = []
            face_embeddings = []
            
            for face_idx, face_box in enumerate(faces):
                # 1.1 Distance Filter: Ignore very small faces (far away)
                f_h = face_box[3] - face_box[1]
                if f_h < (h_proc * 0.05):
                    face_embeddings.append(None)
                    dist_matrix.append([1.0] * len(galleries))
                    continue

                x1, y1, x2, y2 = face_box
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Capture DUAL embeddings
                    spatial_emb = self.recognizer.get_spatial_embedding(roi)
                    vlogger_emb = self.recognizer.get_vlogger_embedding(roi)
                    face_embeddings.append({"spatial": spatial_emb, "vlogger": vlogger_emb})
                    
                    dists = []
                    for g_idx, profile_data in enumerate(galleries):
                        # 1.2 Dual Distance Logic
                        # Spatial distance for tracking consistency (Original Model)
                        spatial_profile = [p["spatial"] for p in profile_data]
                        d_spatial = self.recognizer.mean_top_k_distance(spatial_emb, spatial_profile, k=3)
                        
                        # Vlogger distance for identity confirmation (MobileFaceNet)
                        vlogger_profile = [p["vlogger"] for p in profile_data]
                        d_vlogger = self.recognizer.mean_top_k_distance(vlogger_emb, vlogger_profile, k=5)
                        
                        # Blend: Spatial gives hint, Vlogger confirms
                        # We use spatial for the matrix to keep tracking stable
                        dists.append(d_spatial)
                    dist_matrix.append(dists)
                else:
                    face_embeddings.append(None)
                    dist_matrix.append([1.0] * len(galleries))

            # 2. Global Identity Resolution (Winner-Take-All)
            # Find best match for each profile slot
            for p_idx in range(len(galleries)):
                best_face_idx = -1
                min_d = 1.0
                
                for f_idx in range(len(faces)):
                    if f_idx in current_assignments: continue
                    d = dist_matrix[f_idx][p_idx]
                    
                    # Spatial Consistency Bonus
                    if p_idx in track_states:
                        last_box = track_states[p_idx]["box"]
                        if self.calculate_iou(last_box, faces[f_idx]) > 0.4:
                            d -= 0.1 # Priority to spatially consistent faces
                    
                    if d < min_d:
                        min_d = d
                        best_face_idx = f_idx
                
                # Update Tracking State for this Profile
                if p_idx not in track_states:
                    track_states[p_idx] = {"box": None, "score": 0, "missed": 0, "active": False, "lock_timer": 0}
                
                # Dual Threshold Logic:
                # - Spatial < 0.45 for general tracking consistency
                # - Vlogger < 0.35 for strict identity lock (MobileFaceNet)
                is_match = (best_face_idx != -1 and dist_matrix[best_face_idx][p_idx] < 0.45)
                
                if is_match:
                    current_assignments[best_face_idx] = p_idx
                    target_box = faces[best_face_idx]
                    track_states[p_idx]["box"] = target_box
                    track_states[p_idx]["missed"] = 0
                    
                    # Compute Vlogger-specific distance for identity score
                    v_emb = face_embeddings[best_face_idx]["vlogger"]
                    v_profile = [p["vlogger"] for p in galleries[p_idx]]
                    v_dist = self.recognizer.mean_top_k_distance(v_emb, v_profile, k=5)
                    
                    # Update Integrity Score with High Momentum based on MobileFaceNet
                    if v_dist < 0.35: 
                        track_states[p_idx]["score"] = min(100, track_states[p_idx]["score"] + 30)
                    elif v_dist < 0.45:
                        track_states[p_idx]["score"] = min(100, track_states[p_idx]["score"] + 10)
                    else:
                        track_states[p_idx]["score"] = max(0, track_states[p_idx]["score"] - 5)
                else:
                    # Identity Lost in this frame (Head Tilt or Blur)
                    # Use very slow decay if we were just locked
                    decay = 2 if track_states[p_idx]["active"] else 15
                    track_states[p_idx]["score"] = max(0, track_states[p_idx]["score"] - decay)
                    track_states[p_idx]["missed"] += 1
                    
                    # 2. SPATIAL TRUST (The "Trust Your Eyes" Logic)
                    # If a face is right where the vlogger was, we trust it 100% for stabilization
                    if track_states[p_idx]["box"] is not None:
                        for f_idx, f_box in enumerate(faces):
                            if f_idx not in current_assignments and self.calculate_iou(track_states[p_idx]["box"], f_box) > 0.65:
                                current_assignments[f_idx] = p_idx
                                track_states[p_idx]["box"] = f_box
                                track_states[p_idx]["missed"] = 0
                                # Keeps identity alive during fast movement
                                track_states[p_idx]["score"] = max(1, track_states[p_idx]["score"] - 1)
                                break

                # 3. HYSTERESIS + GRACE PERIOD (Unbreakable Lock)
                # Once locked, we grant a 90-frame (~3s) Grace Period for total stability
                if not track_states[p_idx]["active"]:
                    if track_states[p_idx]["score"] >= 80:
                        track_states[p_idx]["active"] = True
                        track_states[p_idx]["lock_timer"] = 90 # 3 Seconds of Shield
                else:
                    if track_states[p_idx]["lock_timer"] > 0:
                        track_states[p_idx]["lock_timer"] -= 1
                    
                    # Much slower decay when locked
                    if best_face_idx == -1: # No direct AI confirmation
                         track_states[p_idx]["score"] = max(1, track_states[p_idx]["score"] - 1)
                    
                    # NEVER blur if inside grace period OR score is decent
                    if track_states[p_idx]["lock_timer"] <= 0 and track_states[p_idx]["score"] < 25:
                        track_states[p_idx]["active"] = False

                if track_states[p_idx]["active"] and track_states[p_idx]["box"] is not None:
                    vlogger_map[p_idx] = track_states[p_idx]["box"]
                    vlogger_scores[p_idx] = track_states[p_idx]["score"]

            self.result.update_resolution(vlogger_map, vlogger_scores)

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
        """Merges video and audio using FFmpeg with H.264 encoding."""
        print(f"Muxing {output_path}...")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Check if audio exists, fallback to video only if missing
        has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 100
        
        cmd = [ffmpeg, "-y", "-hide_banner"]
        cmd += ["-i", video_path]
        if has_audio:
            cmd += ["-i", audio_path]
        
        # libx264 is much more compatible with Windows players than XVID copy
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23"]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        
        cmd += ["-shortest", output_path]
        
        try:
            # Log errors to file instead of DEVNULL
            log_file = "logs/ffmpeg_mux.log"
            with open(log_file, "w") as f:
                result = subprocess.run(cmd, stdout=f, stderr=f)
            
            if result.returncode == 0:
                if os.path.exists(video_path): os.remove(video_path)
                if os.path.exists(audio_path): os.remove(audio_path)
                print(f"Success: {output_path}")
            else:
                print(f"Mux Failed (Code {result.returncode}). See {log_file}")
        except Exception as e:
            print(f"Merge Critical Error: {e}")

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

    def capture_photo(self, frame_1080p):
        """Saves the high-resolution 1080p frame (with privacy blur applied)."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recordings_dir, f"snap_{ts}.jpg")
        cv2.imwrite(filename, frame_1080p)
        print(f"Photo saved (1080p): {filename}")

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
                try: 
                    if sys.platform == 'win32':
                        os.startfile(os.path.abspath(self.recordings_dir))
                    else:
                        print(f"[UI] Gallery folder: {os.path.abspath(self.recordings_dir)}")
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
                    self.capture_photo(frame)
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
                if self.load_profiles(): 
                    self.state = "VLOGGING"
                else: 
                    self.state = "ENROLLING"
                # Ensure inference loop knows state changed
                time.sleep(0.1)
            elif self.state == "ENROLLING":
                if self.add_profile():
                    self.state = "VLOGGING"
                    print("[STATE] Switched to VLOGGING after successful enrollment.")
                else:
                    self.state = "VLOGGING" if self.vlogger_galleries else "STARTUP"
                    print(f"[STATE] Enrollment failed/canceled. Returning to {self.state}")
                
                # Re-establish main window after enrollment process closes its own
                cv2.namedWindow(self.gui.window_name)
                cv2.setMouseCallback(self.gui.window_name, self.gui._mouse_callback)
            elif self.state in ["VLOGGING", "MENU"]:
                # 1. INSTANT DETECTION in main thread (no lag!)
                faces = self.detector.detect_faces(frame)
                
                # 2. Push for background identity resolution
                self.result.push(frame, faces)
                
                # 3. Get latest vlogger resolution mapping
                v_map, v_scores = self.result.get_resolution()
                
                # 4. Determine which detected faces are vloggers (Shadow Tracking)
                # We unblur if a face is EXACTLY a vlogger or has high IoU with a resolved vlogger box
                vlogger_face_indices = set()
                for i, face_box in enumerate(faces):
                    # Check against all resolved vlogger slots
                    for slot_idx, v_box in v_map.items():
                        if self.calculate_iou(face_box, v_box) > 0.7:
                            vlogger_face_indices.add(i)
                            break

                # 5. Apply Privacy Blur
                if self.privacy_enabled:
                    for i, (x1, y1, x2, y2) in enumerate(faces):
                        if i not in vlogger_face_indices:
                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0: frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
                
                # 6. Synchronous Video Recording (CFR)
                if self.v_recorder is not None:
                    # Write EVERY frame to maintain 30fps timing relative to the file header
                    self.v_recorder.write(frame)
                
                # 6. Create Rendering Canvas with safety checks
                h_proc, w_proc = frame.shape[:2]
                if win_h < 100 or win_w < 100:
                    win_w, win_h = 1280, 720
                
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                
                # Letterboxing / Centering Logic
                v_aspect = 16/9
                if win_w / win_h > v_aspect:
                    v_h = win_h
                    v_w = int(v_h * v_aspect)
                else:
                    v_w = win_w
                    v_h = int(v_w / v_aspect)
                
                ox, oy = (win_w - v_w) // 2, (win_h - v_h) // 2
                self.video_rect = (ox, oy, v_w, v_h)
                
                resized_video = cv2.resize(frame, (v_w, v_h), interpolation=cv2.INTER_LINEAR)
                canvas[oy:oy+v_h, ox:ox+v_w] = resized_video
                
                # 7. Draw Face HUD Elements
                scale_x, scale_y = v_w / w_proc, v_h / h_proc
                
                if self.privacy_enabled:
                    for i, (x1, y1, x2, y2) in enumerate(faces):
                        dx1, dy1 = int(ox + x1 * scale_x), int(oy + y1 * scale_y)
                        dx2, dy2 = int(ox + x2 * scale_x), int(oy + y2 * scale_y)
                        
                        if i not in vlogger_face_indices:
                            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 0, 255), 1)
                            cv2.putText(canvas, "ANONYMIZED", (dx1, dy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        else:
                            # Premium Corner Brackets for Verified Vlogger
                            color, l, t = (0, 255, 0), int(20 * scale_y), 2
                            # Top-Left
                            cv2.line(canvas, (dx1,dy1), (dx1+l,dy1), color, t)
                            cv2.line(canvas, (dx1,dy1), (dx1,dy1+l), color, t)
                            # Top-Right
                            cv2.line(canvas, (dx2,dy1), (dx2-l,dy1), color, t)
                            cv2.line(canvas, (dx2,dy1), (dx2,dy1+l), color, t)
                            # Bottom-Left
                            cv2.line(canvas, (dx1,dy2), (dx1+l,dy2), color, t)
                            cv2.line(canvas, (dx1,dy2), (dx1,dy2-l), color, t)
                            # Bottom-Right
                            cv2.line(canvas, (dx2,dy2), (dx2-l,dy2), color, t)
                            cv2.line(canvas, (dx2,dy2), (dx2,dy2-l), color, t)
                            
                            cv2.putText(canvas, "VLOGGER VERIFIED", (dx1, dy1-15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                
                # 8. Draw UI Overlays
                self.draw_hud(canvas, len(faces), fps)
                self.draw_record_button(canvas)
                if self.state == "MENU" or self.quality_menu_open: 
                    self.draw_menu(canvas)
                
                self.gui.show_frame(canvas)
                
                # 9. Performance Logging
                if time.time() - self.last_log_time > 1.0:
                    avg_score = sum(v_scores.values()) / len(v_scores) if v_scores else 0
                    self.perf_logger.log(len(faces), len(v_scores), avg_score, self.privacy_enabled)
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
