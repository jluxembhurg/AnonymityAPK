"""
Anonymity package — Vlogger-Guard face anonymity app.

This __init__.py adds the package directory to sys.path so that all
flat-style imports (e.g. `from detector import FaceDetector`) continue
to work after the source files are copied into this package by the CI.
"""
import os
import sys

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
