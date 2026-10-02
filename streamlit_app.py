"""
Streamlit Community Cloud root entrypoint for SmartTriage.
Executes ui/app.py while ensuring the root workspace is correctly in sys.path.
"""

from pathlib import Path
import runpy
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute main UI application
ui_script = PROJECT_ROOT / "ui" / "app.py"
runpy.run_path(str(ui_script), run_name="__main__")
