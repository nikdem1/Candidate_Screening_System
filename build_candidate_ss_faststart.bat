@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

echo [1/3] Ensuring dependencies...
python -m pip install -r dev-requirements.txt
python -m pip install nuitka ordered-set zstandard dill

echo [2/3] Building fast-start onedir CandidateSS...
python -m nuitka ^
  --standalone ^
  --assume-yes-for-downloads ^
  --windows-console-mode=disable ^
  --no-deployment-flag=excluded-module-usage ^
  --enable-plugin=tk-inter ^
  --include-module=dill ^
  --include-package-data=whisper ^
  --output-filename=CandidateSS.exe ^
  --output-dir=dist ^
  candidate_ss_gui.py

if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo [3/3] Done. Run:
echo dist\candidate_ss_gui.dist\CandidateSS.exe
exit /b 0
