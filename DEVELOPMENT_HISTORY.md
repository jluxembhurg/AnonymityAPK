# Vlogger-Guard: Development History & Milestone Log

This document tracks the evolution of the Vlogger-Guard project, from its inception to its current state as a professional-grade privacy tool.

## Phase 1: Foundation & Brainstorming (The Identity)
- **Problem Statement**: Vloggers need a way to protect the privacy of bystanders in real-time without blurring themselves.
- **Model Selection**: Evaluated various detectors. Chose **YOLOv8-face (ONNX version)** for its superior performance on consumer CPUs and reliable face detection even in varied lighting.
- **Dependency Strategy**: Committed to a "Lightweight & CPU-First" approach, avoiding heavy CUDA dependencies to ensure accessibility on laptops.

## Phase 2: Core Infrastructure & Logic
- **Modular Design**: Separated concerns into specialized modules: 
  - `detector.py`: Handled the ONNX inference logic.
  - `recognizer.py`: Managed facial embeddings and distance thresholds.
  - `gui.py`: Built high-performance frame capture and display using threading.
- **The "Vlogger Exception"**: Developed the core logic that compares every detected face against an enrollment gallery. Confirmed faces are skipped; all others are instantly blurred.

## Phase 3: Identity & Enrollment
- **Hemisphere Coverage**: Created the `enrollment.py` module. 
- **Guidance System**: Implemented a screen-relative compass to guide the vlogger through a 180° scan (C, N, NE, E, SE, S, SW, W, NW) for reliable recognition from any angle.

## Phase 4: UI Design & HUD Evolution
- **Aesthetic Refinement**: Moved from standard OpenCV windows to a premium HUD with semi-transparent ribbons.
- **Interactive Control**: Added the **Privacy Shield**, **Quality Selector**, and **Recording Controls**.
- **Mode Switching**: Integrated seamless switching between Photo (Snap) and Video (Rec) modes.

## Phase 5: Multimedia & Performance
- **Recording Sync**: Solved the challenge of merging high-FPS video with audio using `ffmpeg` muxing.
- **Thread Optimization**: Isolated inference, UI, and Recording into separate threads to maintain a buttery-smooth 30+ FPS.

## Phase 6: Final Polish: The "Pro" Look
- **Aspect Ratio Correction**: Forced native 1920x1080 (16:9) camera property settings before stream initialization.
- **Letterboxing**: Implemented a centered-video canvas with black bars to prevent stretching on any window size.
- **Dynamic UI Scaling**: Built a resolution-independent rendering engine. All text, icons, and menus now use a 720p-reference scaling factor, keeping the UI sharp and consistent whether processing at 320p or 1080p.
- **Hitbox Synchronization**: Final refined logic to lock ribbons at the absolute window edges and synchronize click detection with the dynamic display layout.

---
*Last Updated: February 28, 2026*
