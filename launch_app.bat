@echo off

REM F1 Championship Predictor - Windows Launch Script
REM Streamlit Web Application Launcher

echo.
echo F1 CHAMPIONSHIP PREDICTOR
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
echo Python is not installed. Please install Python 3.8+ first.
pause
exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
echo pip is not installed. Please install pip first.
pause
exit /b 1
)

echo Installing required packages...
echo.

REM Install requirements
pip install -r requirements.txt

echo.
echo Starting Streamlit application...
echo The app will open in your default web browser
echo Usually available at: http://localhost:8501
echo.
echo ⏹ Press Ctrl+C to stop the application
echo.

REM Launch Streamlit app
streamlit run streamlit_app.py

echo.
echo Thanks for using F1 Championship Predictor!
pause