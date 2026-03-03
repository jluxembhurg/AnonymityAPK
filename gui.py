import cv2
import threading
import time
import numpy as np
import webview
from typing import Optional, Union, Callable

class VideoStream:
    """
    Handles camera capture in a separate thread to maintain UI responsiveness.
    """
    def __init__(self, src: Union[int, str] = 0, width: int = 1920, height: int = 1080):
        self.stream = cv2.VideoCapture(src)
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
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, self.frame

    def set(self, propId, value):
        return self.stream.set(propId, value)

    def stop(self):
        self.stopped = True
        self.stream.release()

class WebViewGUI:
    """
    Manages the standalone pywebview window for the "Anonymity" app.
    """
    def __init__(self, url: str = "http://localhost:8000", title: str = "Anonymity"):
        self.url = url
        self.title = title
        self.window = None
        self.should_exit = False

    def start(self, on_ready: Optional[Callable] = None):
        """Starts the pywebview window."""
        self.window = webview.create_window(self.title, self.url, width=1280, height=720)
        self.window.events.closed += self._on_closed
        webview.start(on_ready)

    def _on_closed(self):
        self.should_exit = True

    def check_exit(self) -> bool:
        return self.should_exit

    def close(self):
        if self.window:
            self.window.destroy()

class VloggerGuardGUI:
    """
    LEGACY: Manages the OpenCV UI window (kept for backward compatibility during transition).
    """
    def __init__(self, window_name: str = "Vlogger-Guard"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.last_click: Optional[tuple[int, int]] = None
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_click = (x, y)

    def get_click(self) -> Optional[tuple[int, int]]:
        click = self.last_click
        self.last_click = None
        return click

    def show_frame(self, frame: np.ndarray):
        cv2.imshow(self.window_name, frame)

    def check_exit(self) -> bool:
        if cv2.waitKey(1) & 0xFF == ord('q'): return True
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1: return True
        except: return True
        return False

    def close(self):
        cv2.destroyAllWindows()
