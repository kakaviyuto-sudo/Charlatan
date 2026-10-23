# -*- coding: utf-8 -*-
"""
YouTube URL -> MP3 standalone helper.
Use only for content you own or are authorized to download.
"""
import os
import sys
import threading
import queue
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import shutil

# Prefer the Python 3.13 interpreter where yt-dlp was confirmed installed.
PREFERRED_PYTHON = Path(r"C:\Users\kakav\AppData\Local\Programs\Python\Python313\python.exe")
if (
    os.name == "nt"
    and PREFERRED_PYTHON.is_file()
    and Path(sys.executable).resolve().as_posix().lower()
        != PREFERRED_PYTHON.resolve().as_posix().lower()
    and os.environ.get("YOUTUBE_MP3_PYTHON_RELAUNCHED") != "1"
):
    env = os.environ.copy()
    env["YOUTUBE_MP3_PYTHON_RELAUNCHED"] = "1"
    subprocess.Popen([str(PREFERRED_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], env=env)
    raise SystemExit(0)

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(r"C:\Users\kakav\neuro_ai\knowledge\style_audio")
MAX_MP3_FILES = 10

def get_python():
    return sys.executable

def safe_count_mp3():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sum(1 for p in OUTPUT_DIR.glob("*.mp3") if p.is_file())

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube URL → MP3 保存ツール")
        self.root.geometry("760x520")
        self.root.minsize(650, 450)
        self.messages = queue.Queue()
        self.busy = False

        main = ttk.Frame(root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="YouTube URL → MP3 保存ツール",
                  font=("", 16, "bold")).pack(anchor="w")
        ttk.Label(main, text="権利者の許可がある音声、またはダウンロードが認められたコンテンツに使用してください。",
                  wraplength=700).pack(anchor="w", pady=(5, 12))

        ttk.Label(main, text="YouTube URL").pack(anchor="w")
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(main, textvariable=self.url_var)
        self.url_entry.pack(fill="x", pady=(4, 10))
        self.url_entry.focus_set()

        row = ttk.Frame(main)
        row.pack(fill="x")
        self.download_btn = ttk.Button(row, text="MP3に変換して保存", command=self.start_download)
        self.download_btn.pack(side="left")
        ttk.Button(row, text="保存フォルダーを開く", command=self.open_folder).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="保存数を更新", command=self.update_count).pack(side="left", padx=(8, 0))

        self.count_var = tk.StringVar()
        ttk.Label(main, textvariable=self.count_var).pack(anchor="w", pady=(10, 8))

        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 8))

        ttk.Label(main, text="処理ログ").pack(anchor="w")
        self.log = tk.Text(main, height=14, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(4, 0))

        self.write_log(f"Python: {sys.executable}")
        self.write_log(f"保存先: {OUTPUT_DIR}")
        self.write_log(f"上限: MP3 {MAX_MP3_FILES} 件（既存ファイルも数えます）")
        self.update_count()
        self.root.after(150, self.poll_messages)

    def write_log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", str(message) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def update_count(self):
        try:
            count = safe_count_mp3()
            self.count_var.set(f"保存済み: {count}/{MAX_MP3_FILES} 件")
        except Exception as e:
            self.count_var.set(f"保存フォルダー確認エラー: {e}")

    def open_folder(self):
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(OUTPUT_DIR))
        except Exception as e:
            messagebox.showerror("フォルダーを開けません", str(e))

    def start_download(self):
        if self.busy:
            return
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning("URLが必要です", "YouTube URLを入力してください。")
            return
        if not re.match(r"^https?://", url, re.IGNORECASE):
            messagebox.showwarning("URLを確認してください", "https:// から始まるURLを入力してください。")
            return
        try:
            count = safe_count_mp3()
        except Exception as e:
            messagebox.showerror("保存先エラー", str(e))
            return
        if count >= MAX_MP3_FILES:
            messagebox.showwarning(
                "上限に達しています",
                f"保存先にMP3がすでに{count}件あります。\n上限は{MAX_MP3_FILES}件です。既存ファイルは自動削除しません。"
            )
            return

        self.busy = True
        self.download_btn.configure(state="disabled")
        self.progress.start(10)
        self.write_log(f"開始: {url}")
        threading.Thread(target=self.download_worker, args=(url,), daemon=True).start()

    def download_worker(self, url):
        try:
            try:
                import yt_dlp
            except ImportError:
                raise RuntimeError(
                    "yt-dlp がインストールされていません。\n"
                    "同梱の setup_and_run.bat を実行して必要ライブラリをインストールしてください。"
                )
            # Prefer a full FFmpeg installation (ffmpeg.exe + ffprobe.exe) on PATH.
            # imageio-ffmpeg bundles ffmpeg.exe, but usually not ffprobe.exe.
            ffmpeg_on_path = shutil.which("ffmpeg")
            ffprobe_on_path = shutil.which("ffprobe")
            if ffmpeg_on_path and ffprobe_on_path:
                ffmpeg_exe = ffmpeg_on_path
                self.messages.put(("log", f"FFmpeg: {ffmpeg_on_path}"))
                self.messages.put(("log", f"FFprobe: {ffprobe_on_path}"))
            else:
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    self.messages.put(("log", f"FFmpeg (imageio-ffmpeg): {ffmpeg_exe}"))
                    if not ffprobe_on_path:
                        self.messages.put(("log", "注意: ffprobe.exe が PATH に見つかりません。MP3変換に失敗する場合は、FFmpeg本体をインストールして ffmpeg.exe と ffprobe.exe の両方を PATH に追加してください。"))
                except ImportError:
                    ffmpeg_exe = ffmpeg_on_path
                    if not ffmpeg_exe or not ffprobe_on_path:
                        raise RuntimeError(
                            "FFmpeg/FFprobe が見つかりません。FFmpegをインストールし、ffmpeg.exe と ffprobe.exe の両方を PATH に追加してください。"
                        )

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            count = safe_count_mp3()
            if count >= MAX_MP3_FILES:
                raise RuntimeError(f"MP3がすでに{count}件あるため、上限{MAX_MP3_FILES}件で停止しました。")

            def logger_hook(message):
                if message:
                    self.messages.put(("log", message))

            def progress_hook(d):
                status = d.get("status")
                if status == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate")
                    got = d.get("downloaded_bytes", 0)
                    if total:
                        pct = got * 100 / total
                        self.messages.put(("log", f"ダウンロード中: {pct:.1f}%"))
                elif status == "finished":
                    self.messages.put(("log", "音声データ取得完了。MP3へ変換しています…"))

            opts = {
                "format": "bestaudio/best",
                "outtmpl": str(OUTPUT_DIR / "%(title).180B [%(id)s].%(ext)s"),
                "noplaylist": True,
                "windowsfilenames": True,
                "restrictfilenames": False,
                "ignoreerrors": False,
                "quiet": True,
                "no_warnings": False,
                "logger": type("Logger", (), {
                    "debug": lambda self, msg: None,
                    "warning": lambda self, msg: logger_hook("警告: " + str(msg)),
                    "error": lambda self, msg: logger_hook("エラー: " + str(msg)),
                })(),
                "progress_hooks": [progress_hook],
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "overwrites": False,
                "continuedl": True,
            }
            if ffmpeg_exe:
                opts["ffmpeg_location"] = str(Path(ffmpeg_exe).parent)

            self.messages.put(("log", "動画情報を確認しています…"))
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise RuntimeError("動画情報を取得できませんでした。URLや公開状態を確認してください。")
                title = info.get("title", "(タイトル不明)")
                video_id = info.get("id", "")
                self.messages.put(("log", f"タイトル: {title}"))
                # Avoid duplicate downloads by video ID in existing filenames.
                if video_id and any(f"[{video_id}]" in p.name for p in OUTPUT_DIR.glob("*.mp3")):
                    raise RuntimeError("この動画IDのMP3は保存先にすでに存在します。重複保存を防ぐため停止しました。")
                ydl.download([url])

            mp3s = list(OUTPUT_DIR.glob("*.mp3"))
            self.messages.put(("log", f"完了: MP3を保存しました。現在 {len(mp3s)}/{MAX_MP3_FILES} 件です。"))
            self.messages.put(("done", True))
        except Exception as e:
            self.messages.put(("log", f"失敗: {e}"))
            self.messages.put(("done", False))

    def poll_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self.write_log(payload)
                elif kind == "done":
                    self.busy = False
                    self.download_btn.configure(state="normal")
                    self.progress.stop()
                    self.update_count()
                    if payload:
                        self.url_var.set("")
        except queue.Empty:
            pass
        self.root.after(150, self.poll_messages)

def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
