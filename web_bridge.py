import cv2
import socketio
import asyncio
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import os
import time

class WebBridge:
    def __init__(self, app_instance, dist_path: str = "anonymity_ui/dist"):
        self.app_instance = app_instance
        self.fastapi_app = FastAPI()
        self.sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
        self.socket_app = socketio.ASGIApp(self.sio, self.fastapi_app)
        self.loop = None
        
        # Static files (React build)
        if os.path.exists(dist_path):
            self.fastapi_app.mount("/assets", StaticFiles(directory=f"{dist_path}/assets"), name="assets")
            @self.fastapi_app.get("/")
            async def read_index():
                index_path = os.path.join(dist_path, "index.html")
                if os.path.exists(index_path):
                    with open(index_path, "r") as f:
                        return Response(content=f.read(), media_type="text/html")
                return Response(content="UI build index.html not found", status_code=404)
        
        self.setup_routes()
        self.setup_socket_events()
        
        self.latest_frame = None
        self.face_metadata = []
        self.frame_lock = threading.Lock()
        self.capture_requested = False
        self.video_writer = None
        os.makedirs("output", exist_ok=True)

    def setup_routes(self):
        @self.fastapi_app.get("/video_feed")
        async def video_feed():
            return StreamingResponse(self.generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    def generate_mjpeg(self):
        """Streaming thread: Applies current AI metadata to latest frame and sends at 30fps."""
        while True:
            frame_to_send = None
            metadata = []
            privacy_active = self.app_instance.privacy_enabled
            capture_flag = False

            # 1. Grab latest data thread-safely
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_send = self.latest_frame.copy()
                    metadata = self.face_metadata.copy()
                if self.capture_requested:
                    capture_flag = True
                    self.capture_requested = False

            if frame_to_send is not None:
                # 2. Apply Blur-on-Stream (The "Seamless" magic)
                # INHIBIT BLUR during enrollment so vlogger can see themselves
                is_enrolling = getattr(self.app_instance, 'state', '') == 'ENROLLING'
                
                if privacy_active and not is_enrolling:
                    has_profile = len(getattr(self.app_instance, 'vlogger_galleries', [])) > 0
                    for face in metadata:
                        if not has_profile or not face.get("isVlogger", False):
                            x, y, w, h = face['x'], face['y'], face['w'], face['h']
                            # Clamp coordinates to frame boundaries
                            fh, fw = frame_to_send.shape[:2]
                            x1, y1 = max(0, x), max(0, y)
                            x2, y2 = min(fw, x+w), min(fh, y+h)
                            
                            roi = frame_to_send[y1:y2, x1:x2]
                            if roi.size > 0:
                                # Quick down-blur-up strategy
                                s_roi = cv2.resize(roi, (roi.shape[1]//4, roi.shape[0]//4))
                                b_roi = cv2.GaussianBlur(s_roi, (25, 25), 0)
                                frame_to_send[y1:y2, x1:x2] = cv2.resize(b_roi, (roi.shape[1], roi.shape[0]))

                # 3. Derive save frame (portrait crop if needed)
                orientation = getattr(self.app_instance, 'orientation', 'landscape')
                if orientation == 'portrait':
                    fh, fw = frame_to_send.shape[:2]
                    crop_w = int(fh * 9 / 16)     # e.g. 720 * 9/16 = 405
                    start_x = (fw - crop_w) // 2   # center the crop
                    save_frame = frame_to_send[:, start_x:start_x + crop_w]
                else:
                    save_frame = frame_to_send

                # 4. Handle photo capture
                if capture_flag:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"output/capture_{timestamp}.jpg"
                    cv2.imwrite(filename, save_frame)
                    print(f"[ACTION] Photo saved: {filename} ({save_frame.shape[1]}x{save_frame.shape[0]})")

                # 5. Handle video recording
                is_recording = getattr(self.app_instance, 'is_recording', False)
                if is_recording:
                    if self.video_writer is None:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"output/record_{timestamp}.mp4"
                        h_out, w_out = save_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (w_out, h_out))
                        print(f"[ACTION] Recording started: {filename} ({w_out}x{h_out})")
                    self.video_writer.write(save_frame)
                else:
                    if self.video_writer is not None:
                        self.video_writer.release()
                        self.video_writer = None
                        print("[ACTION] Recording saved.")

                # 6. Encode and Yield (always stream the full 16:9 frame)
                ret, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Target ~30 FPS for the stream itself
            time.sleep(0.033)

    def update_frame(self, frame):
        """Update the raw frame (from Capture thread)."""
        with self.frame_lock:
            self.latest_frame = frame

    def update_metadata(self, faces):
        """Update the AI face metadata (from AI thread)."""
        with self.frame_lock:
            self.face_metadata = faces

    def setup_socket_events(self):
        @self.sio.on('connect')
        async def connect(sid, environ):
            if self.loop is None:
                self.loop = asyncio.get_event_loop()
            print(f"[WEB] Client connected: {sid}")
            await self.sio.emit('status_update', self.app_instance.get_status(), to=sid)

        @self.sio.on('toggle_privacy')
        async def toggle_privacy(sid, enabled):
            self.app_instance.privacy_enabled = enabled
            print(f"[WEB] Privacy toggled: {enabled}")
            await self.sio.emit('status_update', self.app_instance.get_status())

        @self.sio.on('set_mode')
        async def set_mode(sid, mode):
            self.app_instance.mode = mode
            print(f"[WEB] Mode set to: {mode}")

        @self.sio.on('capture')
        async def capture(sid):
            mode = getattr(self.app_instance, 'mode', 'photo')
            if mode == 'video':
                # Toggle recording on/off
                self.app_instance.is_recording = not self.app_instance.is_recording
                state = "STARTED" if self.app_instance.is_recording else "STOPPED"
                print(f"[WEB] Video recording {state}")
                await self.sio.emit('status_update', self.app_instance.get_status())
            else:
                # Photo capture
                self.capture_requested = True
                print(f"[WEB] Photo capture requested by {sid}")

        @self.sio.on('toggle_orientation')
        async def toggle_orientation(sid):
            self.app_instance.orientation = "landscape" if self.app_instance.orientation == "portrait" else "portrait"
            await self.sio.emit('status_update', self.app_instance.get_status())

        @self.sio.on('start_enrollment')
        async def start_enrollment(sid):
            self.app_instance.start_manual_enrollment()
            await self.sio.emit('status_update', self.app_instance.get_status())

        @self.sio.on('delete_profile')
        async def delete_profile(sid):
            self.app_instance.delete_profile()
            await self.sio.emit('status_update', self.app_instance.get_status())

        @self.sio.on('enroll_retake')
        async def enroll_retake(sid):
            if hasattr(self.app_instance, 'enrollment'):
                self.app_instance.enrollment.reset()
                print("[WEB] Enrollment Reset (Retake)")
                await self.sio.emit('status_update', self.app_instance.get_status())

        @self.sio.on('enroll_cancel')
        async def enroll_cancel(sid):
            self.app_instance.state = "VLOGGING"
            print("[WEB] Enrollment Cancelled")
            await self.sio.emit('status_update', self.app_instance.get_status())

        @self.sio.on('open_gallery')
        async def open_gallery(sid):
            # Natively open the output directory in Windows File Explorer
            output_dir = os.path.abspath("output")
            os.makedirs(output_dir, exist_ok=True)
            os.startfile(output_dir)
            print("[WEB] Gallery folder opened in OS")

    async def emit_status(self):
        await self.sio.emit('status_update', self.app_instance.get_status())

    async def emit_enrollment(self, data):
        await self.sio.emit('enrollment_update', data)

    def emit_status_threadsafe(self):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.emit_status(), self.loop)

    def emit_enrollment_threadsafe(self, data):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.emit_enrollment(data), self.loop)

    def run_server(self, host="localhost", port=8000):
        uvicorn.run(self.socket_app, host=host, port=port, log_level="warning")

    def start_background(self):
        threading.Thread(target=self.run_server, daemon=True).start()
