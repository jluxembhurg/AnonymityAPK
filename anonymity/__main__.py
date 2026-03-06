"""
Briefcase entry point for the Anonymity Android app.
The CI copies all flat .py modules into this package before building,
so all imports resolve correctly via __init__.py's sys.path setup.
"""
import logging

# Trigger __init__.py sys.path setup before any other imports
import anonymity  # noqa: F401 (side-effect import)

from main_android import AndroidAnonymityApp

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = AndroidAnonymityApp()
    app.run()
