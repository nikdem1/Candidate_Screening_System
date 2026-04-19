@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

set APP_MODE=
if exist "dist\candidate_ss_gui.dist\CandidateSS.exe" (
  set APP_MODE=onedir
) else if exist "dist\CandidateSS.exe" (
  set APP_MODE=onefile
)

if "%APP_MODE%"=="" (
  echo No built executable found in dist.
  echo Building fast-start onedir package...
  call build_candidate_ss_faststart.bat
  if errorlevel 1 exit /b 1
  if exist "dist\candidate_ss_gui.dist\CandidateSS.exe" (
    set APP_MODE=onedir
  ) else if exist "dist\CandidateSS.exe" (
    set APP_MODE=onefile
  )
)

if "%APP_MODE%"=="" (
  echo Build completed but executable was not found.
  exit /b 1
)

echo Detected build mode: %APP_MODE%

set ISCC_EXE=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if not exist "%ISCC_EXE%" set ISCC_EXE=%ProgramFiles%\Inno Setup 6\ISCC.exe

if not exist "%ISCC_EXE%" (
  echo Inno Setup compiler not found.
  echo Install Inno Setup 6: https://jrsoftware.org/isdl.php
  exit /b 1
)

echo Building installer...
if /I "%APP_MODE%"=="onedir" (
  "%ISCC_EXE%" /DAppBuildMode=onedir "installer\CandidateSS.iss"
) else (
  "%ISCC_EXE%" /DAppBuildMode=onefile "installer\CandidateSS.iss"
)

if errorlevel 1 (
  echo Installer build failed.
  exit /b 1
)

echo Done: dist\CandidateSS_Installer.exe
exit /b 0
