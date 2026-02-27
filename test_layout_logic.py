import numpy as np
import cv2
import os
import sys

# Add current dir to path to import local modules
sys.path.append(os.getcwd())

from unittest.mock import MagicMock

# Mocking modules before importing main
import cv2
import sounddevice as sd
sys.modules['cv2'] = MagicMock()
sys.modules['sounddevice'] = MagicMock()
sys.modules['scipy.io.wavfile'] = MagicMock()
sys.modules['imageio_ffmpeg'] = MagicMock()
sys.modules['gui'] = MagicMock()
sys.modules['detector'] = MagicMock()
sys.modules['recognizer'] = MagicMock()
sys.modules['enrollment'] = MagicMock()

from main import VloggerGuardApp

def test_layouts():
    # Patch __init__ to avoid hardware setup but keep self.regions etc.
    original_init = VloggerGuardApp.__init__
    def mock_init(self):
        self.vlogger_galleries = []
        self.regions = {}
    
    VloggerGuardApp.__init__ = mock_init
    app = VloggerGuardApp()
    
    # Manually restore update_layout since it's the one we want to test
    from main import VloggerGuardApp as OriginalClass
    app.update_layout = OriginalClass.update_layout.__get__(app, OriginalClass)
    
    resolutions = [
        (320, 568),   # Portrait 320p
        (1280, 720),  # Landscape 720p
        (1920, 1080), # Landscape 1080p
        (1080, 1920)  # Portrait 1080p
    ]
    
    for w, h in resolutions:
        print(f"Testing {w}x{h}...")
        app.update_layout(w, h)
        
        # Check some key regions
        px, py, pw, ph = app.regions["profile_icon"]
        rx, ry, rw, rh = app.regions["record_btn"]
        is_portrait = h > w
        
        # Verify alignment
        if is_portrait:
            # Profile icon should be centered at top
            expected_px = w // 2 - 45
            if px != expected_px:
                print(f"FAILED Portrait Profile Icon X: got {px}, expected {expected_px}")
                return False
        else:
            # Landscape Profile should be at 10, 5
            if px != 10:
                print(f"FAILED Landscape Profile Icon X: got {px}, expected 10")
                return False
                
        print(f"  OK: {'Portrait' if is_portrait else 'Landscape'}")

    print("\nAll layout calculations verified!")
    return True

if __name__ == "__main__":
    if test_layouts():
        sys.exit(0)
    else:
        sys.exit(1)
