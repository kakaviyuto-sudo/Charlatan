@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title MP3専用 話し方学習

set "SCRIPT=%~dp0MP3とTxtから話し方を自動学習_分離対応.py"
set "PYTHON_EXE="

echo ========================================
echo MP3専用 話し方学習
echo ========================================
echo.
if not exist "%SCRIPT%" (
  echo [ERROR] 学習用Pythonファイルが見つかりません:
  echo "%SCRIPT%"
  goto ERROR_END
)

if exist "%~dp0main_venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0main_venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0.venv313\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv313\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\.venv313\Scripts\python.exe" set "PYTHON_EXE=%USERPROFILE%\.venv313\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if defined PYTHON_EXE (
  "%PYTHON_EXE%" "%SCRIPT%" --mode mp3
) else (
  where py >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python が見つかりません。
    goto ERROR_END
  )
  py -3.13 "%SCRIPT%" --mode mp3
)
set "RESULT=%ERRORLEVEL%"
goto FINISH

:ERROR_END
set "RESULT=1"
:FINISH
echo.
if "%RESULT%"=="0" (echo [完了] MP3学習処理が終了しました。) else (echo [ERROR] 終了コード: %RESULT%)
pause
endlocal & exit /b %RESULT%
