import cv2
import threading
import time
import numpy as np
from typing import Optional, Union

class VideoStream:
    """
    Handles camera capture in a separate thread to maintain UI responsiveness.
    """
    def __init__(self, src: Union[int, str] = 0, width: int = 1920, height: int = 1080):
        self.stream = cv2.VideoCapture(src)
        # Set properties BEFORE starting thread or reading
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.stream.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def set(self, propId, value):
        return self.stream.set(propId, value)

    def stop(self):
        self.stopped = True
        self.stream.release()

    def release(self):
        self.stop()

class VloggerGuardGUI:
    """
    Manages the OpenCV UI window, frame display, and mouse interactions.
    """
    def __init__(self, window_name: str = "Vlogger-Guard"):
        self.window_name = window_name
        # Allow window resizing
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.last_click: Optional[tuple[int, int]] = None
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)

    def get_click(self) -> Optional[tuple[int, int]]:
        """Returns the last click coordinates and resets them."""
        click = self.last_click
        self.last_click = None
        return click

    def get_window_size(self) -> tuple[int, int]:
        """Returns the current window (width, height) using getWindowImageRect for accuracy."""
        try:
            rect = cv2.getWindowImageRect(self.window_name)
            if rect and len(rect) == 4:
                return int(rect[2]), int(rect[3])
            return 1280, 720
        except:
            return 1280, 720

    def show_frame(self, frame: np.ndarray):
        """Displays a frame in the UI."""
        cv2.imshow(self.window_name, frame)

    def check_exit(self) -> bool:
        """Checks if the user pressed 'q' or closed the window via 'X'."""
        if cv2.waitKey(1) & 0xFF == ord('q'): return True
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1: return True
        except: return True
        return False

    def close(self):
        cv2.destroyAllWindows()
