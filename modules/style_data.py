"""Neuro AI style-data file helpers (phase 1 extraction).

This module only reads/writes the existing JSON files. It does not import the
main AI, Ollama, audio libraries, or alter the JSON schema.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load_text_style(path: Path, label: str) -> str:
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, str) else ""
    except (OSError, json.JSONDecodeError) as error:
        print(f"{label}の読み込みエラー:", repr(error))
        return ""


def load_prosody_profile(path: Path) -> dict:
    if not path.exists():
        return {"intonation_variability": 1.0, "samples": 0, "videos": 0}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("イントネーションデータの形式が不正です")
        return {
            "intonation_variability": float(data.get("intonation_variability", 1.0)),
            "samples": int(data.get("samples", 0)),
            "videos": int(data.get("videos", 0)),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print("イントネーションデータの読み込みエラー:", repr(error))
        return {"intonation_variability": 1.0, "samples": 0, "videos": 0}


def save_prosody_profile(path: Path, profile: dict) -> bool:
    temp_file = path.with_suffix(".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as handle:
            json.dump(profile, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_file, path)
        return True
    except (OSError, TypeError, ValueError) as error:
        print("イントネーション情報の保存エラー:", repr(error))
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def save_text_style(path: Path, style: str, label: str) -> bool:
    temp_file = path.with_suffix(".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as handle:
            json.dump(style, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_file, path)
        return True
    except (OSError, TypeError, ValueError) as error:
        print(f"{label}の保存エラー:", repr(error))
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def load_style_sources(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as error:
        print("話し方学習元の読み込みエラー:", repr(error))
        return []


def save_style_sources(path: Path, items: list) -> None:
    temp_file = path.with_suffix(".tmp")
    with temp_file.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, ensure_ascii=False, indent=2)
    os.replace(temp_file, path)
