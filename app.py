# F1 Championship Predictor - Simple Entry Point
# This file just redirects to streamlit_app.py for deployment compatibility

if __name__ == "__main__":
    import runpy
    runpy.run_path("streamlit_app.py")