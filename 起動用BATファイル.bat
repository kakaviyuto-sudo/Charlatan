@echo off
chcp 65001 >nul
setlocal
title Neuro AI - SBV2 修正版

cd /d "%~dp0"
set "SCRIPT=%~dp0main_nijisanji_vtube_studio_all_audio_learning_v14_sbv2_fixed.py"
set "PYTHON=C:\Users\kakav\.venv313\Scripts\python.exe"

if not exist "%SCRIPT%" (
  echo [ERROR] 修正版Pythonファイルが見つかりません。
  echo このBATとPythonファイルを同じフォルダーに置いてください。
  echo 確認先: "%SCRIPT%"
  pause
  exit /b 1
)

if not exist "%PYTHON%" set "PYTHON=%~dp0.venv313\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] Pythonが見つかりません。
  echo 確認した場所:
  echo C:\Users\kakav\.venv313\Scripts\python.exe
  echo %~dp0.venv313\Scripts\python.exe
  pause
  exit /b 1
)

echo ================================================
echo Neuro AI / Style-Bert-VITS2 / 字幕ちゃん連携
echo ================================================
echo Python: "%PYTHON%"
"%PYTHON%" --version
echo.
echo 字幕ちゃんはPython起動後に次のURLで開きます:
echo http://127.0.0.1:8765/v2/
echo 状態確認:
echo http://127.0.0.1:8765/health
echo.
echo 音声出力: VB-CABLE + Windows既定のヘッドホン/スピーカー
echo 既定値を変更する場合は環境変数 NEURO_VOICE_OUTPUT_DEVICE /
echo NEURO_VOICE_MONITOR_DEVICE を設定してください。
echo Ctrl+C で終了できます。
echo ================================================
echo.

"%PYTHON%" "%SCRIPT%"
set "EXITCODE=%ERRORLEVEL%"
echo.
echo Pythonが終了しました。終了コード: %EXITCODE%
pause
exit /b %EXITCODE%
