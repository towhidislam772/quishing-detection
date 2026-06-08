@echo off
REM ==== One-click runner for the quishing detection pipeline ====
REM Run from:  C:\Users\USER\Desktop\QR paper\content\quishing-detection\

setlocal
cd /d "%~dp0"

if not exist .venv (
    echo [setup] Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate

echo [setup] Installing requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd src

echo [1/7] Extract data...
python 01_extract_data.py    || goto :err
echo [2/7] Build dataset...
python 02_build_dataset.py   || goto :err
echo [3/7] URL features...
python 03_url_features.py    || goto :err
echo [4/7] Train XGBoost...
python 04_train_xgb.py       || goto :err
echo [5/7] Train CNN...
python 05_train_cnn.py       || goto :err
echo [6/7] Train hybrid...
python 06_train_hybrid.py    || goto :err
echo [7/7] Evaluate + figures...
python 07_evaluate.py        || goto :err

echo.
echo ===== DONE =====
echo Models   : ..\models
echo Results  : ..\results
echo Figures  : ..\figures
exit /b 0

:err
echo *** Pipeline failed at the last step. ***
exit /b 1
