"""Neuro AI text-to-speech engine helpers (phase 2 extraction).

This module keeps synthesis separate from the main application. Playback,
transcript publishing, and VTube Studio hotkeys remain in the main program.
Heavy/optional dependencies are imported lazily or injected by the caller.
"""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Any


def synthesize_edge_audio(
    *, text: str, voice: str, rate: str, pitch: str, volume: str,
    resolve_ffmpeg: Callable[[], str], soundfile_module: Any,
):
    """Generate Edge TTS audio and return (float32 audio, sample_rate)."""
    try:
        import edge_tts
    except ImportError as error:
        raise RuntimeError(
            "edge-tts が入っていないよ。PowerShellで `py -m pip install edge-tts` を実行してね。"
        ) from error

    async def _generate(mp3_path: Path):
        communicator = edge_tts.Communicate(
            text=text, voice=voice, rate=rate, pitch=pitch, volume=volume
        )
        await communicator.save(str(mp3_path))

    with tempfile.TemporaryDirectory(prefix="neuro_edge_tts_") as temp_dir:
        mp3_path = Path(temp_dir) / "speech.mp3"
        wav_path = Path(temp_dir) / "speech.wav"
        asyncio.run(_generate(mp3_path))
        ffmpeg = resolve_ffmpeg()
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(mp3_path), "-vn", "-ac", "1",
                 "-ar", "24000", "-c:a", "pcm_s16le", str(wav_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                check=True, timeout=90,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "音声変換用FFmpegが見つからないよ。`py -m pip install imageio-ffmpeg` を実行してね。"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", errors="replace")[-800:] if error.stderr else ""
            raise RuntimeError(f"edge-tts音声のWAV変換に失敗したよ: {detail}") from error
        audio, sample_rate = soundfile_module.read(str(wav_path), dtype="float32")
        return audio, sample_rate


def synthesize_voicevox_audio(
    *, text: str, session: Any, base_url: str, speaker: int,
    prosody_profile: dict, soundfile_module: Any, bytes_io: Any,
):
    """Generate VOICEVOX audio while preserving the existing tuning values."""
    query_response = session.post(
        f"{base_url}/audio_query", params={"text": text, "speaker": speaker}, timeout=30
    )
    query_response.raise_for_status()
    query = query_response.json()
    query["speedScale"] = 1.05
    query["volumeScale"] = 1.0
    query["pauseLengthScale"] = 0.92
    learned_variability = float(prosody_profile.get("intonation_variability", 1.0))
    intonation_factor = max(0.90, min(1.22, 0.96 + learned_variability * 0.10))
    query["intonationScale"] = max(0.80, min(1.35, 1.08 * intonation_factor))
    response = session.post(
        f"{base_url}/synthesis", params={"speaker": speaker}, json=query, timeout=60
    )
    response.raise_for_status()
    return soundfile_module.read(bytes_io(response.content), dtype="float32")
