import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from detector import FaceDetector
from recognizer import FaceRecognizer
import threading
import os

# Android-specific imports for permissions
try:
    from android.permissions import request_permissions, Permission
    ANDROID = True
except ImportError:
    ANDROID = False

class KivyVloggerGuard(App):
    def build(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.privacy_enabled = True
        self.vlogger_profiles = [] # Load from disk if available
        
        # Layout
        self.layout = BoxLayout(orientation='vertical')
        
        # Header
        self.header = Label(text="VLOGGER GUARD AI (Loading...)", size_hint_y=0.1)
        self.layout.add_widget(self.header)
        
        # Camera Display
        self.img = Image()
        self.layout.add_widget(self.img)
        
        # Controls
        self.controls = BoxLayout(size_hint_y=0.15)
        self.btn_privacy = Button(text="PRIVACY: ON")
        self.btn_privacy.bind(on_press=self.toggle_privacy)
        self.controls.add_widget(self.btn_privacy)
        
        self.btn_enroll = Button(text="ENROLL FACE")
        self.btn_enroll.bind(on_press=self.start_enrollment)
        self.controls.add_widget(self.btn_enroll)
        
        self.layout.add_widget(self.controls)
        
        # Start capture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)
        
        if ANDROID:
            request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE])
            
        return self.layout

    def toggle_privacy(self, instance):
        self.privacy_enabled = not self.privacy_enabled
        instance.text = f"PRIVACY: {'ON' if self.privacy_enabled else 'OFF'}"

    def start_enrollment(self, instance):
        # Simplistic enrollment for the Kivy prototype
        self.header.text = "ENROLLING... (Hold Still)"
        # Future: implement gallery collection here
        pass

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Simple AI Pipeline
        faces = self.detector.detect_faces(frame)
        
        # Apply Blur if needed
        if self.privacy_enabled:
            for (x1, y1, x2, y2) in faces:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert to Kivy Texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture1
        self.header.text = f"VLOGGER GUARD ACTIVE - Detection: {len(faces)} faces"

    def on_stop(self):
        self.capture.release()

if __name__ == '__main__':
    KivyVloggerGuard().run()
