@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title YouTube URL MP3 保存ツール

set "PYTHON=%USERPROFILE%\.venv313\Scripts\python.exe"
if not exist "%PYTHON%" (
  set "PYTHON=%~dp0.venv313\Scripts\python.exe"
)
if not exist "%PYTHON%" (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.13 -m pip install -U yt-dlp imageio-ffmpeg
    if errorlevel 1 goto error
    py -3.13 "%~dp0youtube_mp3_downloader.py"
    goto end
  )
  echo Python 3.13 が見つかりません。
  echo C:\Users\kakav\.venv313\Scripts\python.exe を確認してください。
  pause
  exit /b 1
)

echo 必要なライブラリをインストール／更新しています...
"%PYTHON%" -m pip install -U yt-dlp imageio-ffmpeg
if errorlevel 1 goto error

echo 起動します...
"%PYTHON%" "%~dp0youtube_mp3_downloader.py"
goto end

:error
echo.
echo エラーが発生しました。ネット接続とPython環境を確認してください。
pause
exit /b 1

:end
endlocal
