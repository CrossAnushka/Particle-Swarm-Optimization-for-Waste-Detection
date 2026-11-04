"""Root entry point for Streamlit Community Cloud."""
import runpy
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

if __name__ == "__main__" or "streamlit" in sys.modules:
    runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
