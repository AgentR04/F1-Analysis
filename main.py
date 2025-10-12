# F1 Championship Predictor - Main Entry Point
# Simple redirect to streamlit_app.py for deployment platforms

import runpy
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Execute streamlit_app.py directly 
if __name__ == "__main__":
    runpy.run_path("streamlit_app.py", run_name="__main__")