"""
run_app.py
──────────
Launch the Neuro-CX Streamlit dashboard.

Usage:
  py run_app.py
"""

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
dashboard = os.path.join(ROOT, "src", "app", "dashboard.py")

if __name__ == "__main__":
    print("Launching Neuro-CX dashboard...")
    print(f"Dashboard: {dashboard}")
    print("Open your browser at http://localhost:8501\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard,
        "--server.headless", "true",
        "--theme.base", "dark",
        "--theme.primaryColor", "#667eea",
        "--theme.backgroundColor", "#0d0d1a",
        "--theme.secondaryBackgroundColor", "#16213e",
        "--theme.textColor", "#e0e0e0",
    ], cwd=ROOT)
