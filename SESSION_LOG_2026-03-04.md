# 🛡️ VlogBlur — Session Development Log
**Date:** 2026-03-04  
**Session Time:** ~03:50 – 05:00 PHT  

---

## Accomplishments This Session

### Phase 10 · Grid Shrinking (50% Reduction)
- **Problem:** The interactive 3×3 enrollment grid was too large and was bleeding off the screen on portrait mode.
- **Fix:** Halved both the UI grid dimensions in `App.tsx` (`w-[24%] h-[42%]` → `w-[12%] h-[21%]`) and the matching backend detection thresholds in `enrollment.py` (`h_thresh=0.04` → `0.02`, `v_thresh=0.07` → `0.035`).
- **Result:** The grid now requires only very subtle, precise head movements to complete enrollment.

---

### Phase 10 · MobileFaceNet Threshold Fix
- **Problem:** After upgrading from GhostFaceNet to MobileFaceNet ONNX, the anonymity blur stopped working — strangers were no longer being blurred out.
- **Root Cause:** MobileFaceNet groups face embeddings much closer together than GhostFaceNet. The `v_dist < 0.40` threshold was so loose it classified strangers as the vlogger.
- **Fix:** Tightened the cosine distance threshold in `main_anonymity.py` from `0.40` to `0.25`.
- **Result:** The AI is now correctly skeptical, rejecting strangers and applying `cv2.GaussianBlur` to them.

---

### Phase 11 · Background Thread Crash Fix (Critical)
- **Problem:** After a successful enrollment, face blurring stopped entirely.
- **Root Cause:** `main_anonymity.py` was loading the saved `.npy` profiles as `list(data)` (flat), causing a `TypeError: string indices must be integers` in the recognition background thread — permanently crashing it.
- **Fix:** Changed to `[list(data)]` to correctly wrap the loaded data in a 2D list, restoring the thread's ability to iterate profile records.

---

### Phase 11 · Empty-Profile Force-Blur
- **Feature:** When the user deletes their vlogger profile, all detected faces should instantly be blurred (fail-safe mode).
- **Fix:** Added a `has_profile` check inside the `generate_mjpeg` rendering loop of `web_bridge.py`. If `len(vlogger_galleries) == 0`, the bridge now forces `cv2.GaussianBlur` on every detected bounding box immediately — no waiting for the AI to classify faces as "Unknown."

---

### Phase 12 · Capture & Album Buttons
- **Problem:** The Capture button only printed a debug string. The Album button was hardcoded to `console.log('Gallery clicked')`.
- **Fixes:**
    - **Capture:** Added a `capture_requested` boolean flag to `WebBridge`. When set, the render loop intercepts the **fully blurred** frame and writes it as a timestamped `.jpg` to the local `output/` directory.
    - **Album:** Added an `open_gallery` socket event listener in `web_bridge.py`. When triggered, it calls `os.startfile(output_dir)` to natively open the Windows File Explorer on the output folder.
    - Wired up both `handleCapture` and `handleGalleryClick` in `App.tsx` for both Portrait and Landscape layouts.

---

### Phase 13 · Video Recording (Mode-Aware Capture)
- **Feature:** The REC button in Video mode needs to actually record video, not take a photo.
- **Fixes:**
    - **Mode-Aware Capture:** The `capture` socket handler now checks `self.app_instance.mode`. In `photo` mode it takes a snapshot. In `video` mode it toggles `self.app_instance.is_recording`.
    - **UI Toggle:** Because `is_recording` is now driven by the backend state and broadcast via `status_update`, the React UI's Red Circle (start) ↔ Red Square (stop) toggle works correctly.
    - **VideoWriter:** Added `self.video_writer` to `WebBridge`. When recording starts, a `cv2.VideoWriter` is initialized targeting a timestamped `.mp4` in `output/`. Every blurred frame is written to it. On stop, the writer is released, finalizing the file.
    - **Portrait Crop:** Both photo captures and video recordings apply a center-crop to `405×720` (9:16) when `orientation == 'portrait'`, matching exactly what the user sees on screen.

---

## Files Modified This Session
| File | Description |
|---|---|
| `main_anonymity.py` | Profile loading fix (`[list(data)]`), threshold tightened to `0.25` |
| `enrollment.py` | Grid thresholds halved (`0.02` / `0.035`) |
| `web_bridge.py` | Capture flag, VideoWriter, portrait crop, force-blur on empty profile, `open_gallery` endpoint |
| `anonymity_ui/src/app/App.tsx` | Wired `handleCapture`, `handleGalleryClick`; fixed both Portrait + Landscape layouts |

---

## Repository
**GitHub:** https://github.com/luexmbhurg/Anonymity.git
