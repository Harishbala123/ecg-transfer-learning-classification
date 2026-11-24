@echo off
REM Launch the interactive demo for this project in your browser (http://localhost:8503).
REM First time only: create the venv and install requirements (see README, "Installation").
cd /d "%~dp0"
set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY if exist "..\.venvs\ecg\Scripts\python.exe" set PY=..\.venvs\ecg\Scripts\python.exe
if not defined PY (
  echo No virtual environment found. Run:  py -3.11 -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt streamlit
  pause
  exit /b 1
)
set TF_CPP_MIN_LOG_LEVEL=2
echo Starting ECG apnea demo on http://localhost:8503  (close this window to stop)
"%PY%" -m streamlit run app.py --server.port 8503 --server.headless false --browser.gatherUsageStats false
