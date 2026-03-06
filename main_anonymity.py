import cv2
import numpy as np
import os
import time
import threading
from typing import Optional, List
from detector import FaceDetector
from recognizer import FaceRecognizer
from enrollment import EnrollmentModule
from tracker import FaceTracker
from web_bridge import WebBridge
from gui import VideoStream, WebViewGUI

class AnonymityApp:
    def __init__(self):
        # Models: detector.onnx for bounding boxes, mobilefacenet.onnx for identity
        self.detector = FaceDetector()          # uses models/detector.onnx
        self.recognizer = FaceRecognizer()      # uses models/mobilefacenet.onnx for vlogger ID

        # Camera capture thread (hardware speed)
        self.vs = VideoStream(src=0, width=1280, height=720)
        self.vs.start()

        self.enrollment = EnrollmentModule(self.detector, self.recognizer)

        # Web streaming bridge with blur-on-stream
        self.bridge = WebBridge(self)
        self.gui = WebViewGUI()

        self.vlogger_galleries = []
        self.privacy_enabled = True
        self.mode = 'video'
        self.state = "VLOGGING"
        self.orientation = "portrait"
        self.is_recording = False
        self.face_metadata = []  # Shared with bridge

        self.stop_threads = False
        self.load_profiles()

    def load_profiles(self):
        path = "data/vlogger_profiles.npy"
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                self.vlogger_galleries = [list(data)]
                print(f"[LOAD] {len(self.vlogger_galleries)} profiles loaded.")
                return True
            except:
                print("[ERROR] Corrupt profiles. Resetting.")
                self.vlogger_galleries = []
        return False

    def save_profiles(self):
        os.makedirs("data", exist_ok=True)
        np.save("data/vlogger_profiles.npy", np.array(self.vlogger_galleries, dtype=object))

    def get_status(self):
        return {
            "anonymity_active": self.privacy_enabled,
            "mode": self.mode,
            "is_recording": self.is_recording,
            "state": self.state,
            "orientation": self.orientation,
            "faces": self.face_metadata,
            "has_profile": len(self.vlogger_galleries) > 0
        }

    def delete_profile(self):
        self.vlogger_galleries = []
        path = "data/vlogger_profiles.npy"
        if os.path.exists(path):
            os.remove(path)
        self.state = "VLOGGING"
        self.face_metadata = []
        self.enrollment.reset()
        print("[ACTION] Profile deleted.")

    def start_manual_enrollment(self):
        print("[ACTION] Starting manual enrollment")
        self.state = "ENROLLING"
        self.enrollment.reset()

    def trigger_capture(self):
        print(f"[ACTION] {self.mode} capture triggered")

    def capture_and_stream_loop(self):
        """
        Capture Rhythm: Pushes the latest raw frame to the bridge at hardware speed.
        The bridge handles blurring using the AI thread's latest metadata.
        """
        while not self.stop_threads:
            ret, frame_raw = self.vs.read()
            if ret and frame_raw is not None:
                frame = cv2.flip(frame_raw, 1)
                if frame.shape[0] != 720:
                    frame = cv2.resize(frame, (1280, 720))
                self.bridge.update_frame(frame)
            time.sleep(0.005)  # Push at ~100fps cap to avoid starvation

    def ai_inference_loop(self):
        """
        AI Rhythm: Asynchronously runs face detection and recognition.
        Uses FaceTracker to maintain responsive face positions at 30fps
        BETWEEN heavy detection cycles.
        """
        print("[ENGINE] AI Inference Loop Started")

        tracker = FaceTracker(max_disappeared=20)

        # Identity cache: tracks if each face ID is vlogger (avoids re-running mobilefacenet)
        identity_cache = {}  # track_id -> {"isVlogger": bool, "score": int, "missed": int}

        galleries = self.vlogger_galleries

        frame_count = 0
        DETECT_EVERY = 5      # Run detector.onnx every 5 frames (~6fps on slow CPUs)
        RECOGNIZE_EVERY = 15  # Run mobilefacenet.onnx every 15 frames (~2fps for ID)

        while not self.stop_threads:
            ret, frame_raw = self.vs.read()
            if not ret or frame_raw is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame_raw, 1)
            if frame.shape[0] != 720:
                frame = cv2.resize(frame, (1280, 720))

            frame_count += 1

            # --- ENROLLMENT STATE ---
            if self.state == "ENROLLING":
                status = self.enrollment.process_enrollment_frame(frame)
                self.bridge.emit_enrollment_threadsafe(status)
                if status["is_complete"]:
                    self.load_profiles()
                    galleries = self.vlogger_galleries
                    self.state = "VLOGGING"
                    tracker = FaceTracker(max_disappeared=20)
                    identity_cache = {}
                    self.bridge.emit_status_threadsafe()
                continue

            galleries = self.vlogger_galleries

            # --- DETECTION PHASE (every N frames) ---
            ran_detection = (frame_count % DETECT_EVERY == 0)
            ran_recognition = (frame_count % RECOGNIZE_EVERY == 0)

            detected_boxes = []
            vlogger_indices = set()

            if ran_detection:
                proxy = cv2.resize(frame, (640, 360))
                proxy_faces = self.detector.detect_faces(proxy)
                detected_boxes = [
                    [int(x1 * 2), int(y1 * 2), int(x2 * 2), int(y2 * 2)]
                    for (x1, y1, x2, y2) in proxy_faces
                ]

            # --- RECOGNITION PHASE (MobileFaceNet only when it's time) ---
            if ran_recognition and galleries and detected_boxes:
                for i, box in enumerate(detected_boxes):
                    x1, y1, x2, y2 = box
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    roi_proc = cv2.resize(roi, (112, 112))
                    v_emb = self.recognizer.get_vlogger_embedding(roi_proc)
                    for profile_data in galleries:
                        v_profile = [p["vlogger"] for p in profile_data]
                        v_dist = self.recognizer.mean_top_k_distance(v_emb, v_profile, k=5)
                        if v_dist < 0.25:
                            vlogger_indices.add(i)
                            break

            # --- TRACKER UPDATE ---
            if ran_detection:
                # Re-match detections and refresh optical flow seeds
                metadata = tracker.reinit(
                    frame, detected_boxes,
                    vlogger_indices=list(vlogger_indices),
                    recognition_ran=ran_recognition
                )
            else:
                # Use Lucas-Kanade motion tracking (no AI needed)
                metadata = tracker.tick(frame)

            # Push updated metadata to bridge (bridge applies blur at stream time)
            self.face_metadata = metadata
            self.bridge.update_metadata(metadata)

            # Small yield to prevent starving capture thread
            time.sleep(0.001)

    def run(self):
        self.bridge.start_background()

        # Start Decoupled Rhythms
        threading.Thread(target=self.capture_and_stream_loop, daemon=True).start()
        threading.Thread(target=self.ai_inference_loop, daemon=True).start()

        def status_emitter():
            while not self.stop_threads:
                self.bridge.emit_status_threadsafe()
                time.sleep(2)
        threading.Thread(target=status_emitter, daemon=True).start()

        self.gui.start()

        self.stop_threads = True
        self.vs.stop()

if __name__ == "__main__":
    AnonymityApp().run()
