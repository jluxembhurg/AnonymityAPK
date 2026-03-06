import os
import threading
import time
from main_anonymity import AnonymityApp
from gui import WebViewGUI
import logging

# Android-specific imports
try:
    from android.permissions import request_permissions, Permission
    ANDROID = True
except ImportError:
    ANDROID = False

class AndroidAnonymityApp(AnonymityApp):
    def __init__(self):
        super().__init__()
        self.orientation = "portrait" # Default for mobile
        
    def request_android_permissions(self):
        if ANDROID:
            print("[ANDROID] Requesting permissions...")
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.INTERNET,
                Permission.RECORD_AUDIO
            ])
            # Give some time for permissions to be granted
            time.sleep(2)

    def run(self):
        # 1. Handle Permissions
        self.request_android_permissions()
        
        # 2. Start Backend (FastAPI + AI Loops)
        self.bridge.start_background()
        
        # Start Decoupled Rhythms
        threading.Thread(target=self.capture_and_stream_loop, daemon=True).start()
        threading.Thread(target=self.ai_inference_loop, daemon=True).start()

        def status_emitter():
            while not self.stop_threads:
                self.bridge.emit_status_threadsafe()
                time.sleep(2)
        threading.Thread(target=status_emitter, daemon=True).start()

        # 3. Launch UI
        # On Android, we use the same WebViewGUI but we need to ensure local networking is stable
        print("[ANDROID] Launching WebView...")
        self.gui.start()

        self.stop_threads = True
        self.vs.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = AndroidAnonymityApp()
    app.run()
