"""Knowledge-asset integration layer for the Charlatan runtime.

This keeps the project's existing JSON/TXT/MP3 knowledge files as a single,
loadable runtime bundle without forcing a breaking refactor of the current app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"
STYLE_AUDIO_DIR = KNOWLEDGE_DIR / "style_audio"


@dataclass
class KnowledgeBundle:
    repo_root: Path = REPO_ROOT
    knowledge_dir: Path = KNOWLEDGE_DIR
    style_audio_dir: Path = STYLE_AUDIO_DIR
    style_audio_sources: List[Dict[str, Any]] = field(default_factory=list)
    youtube_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    streamer_style: str = ""
    phrasing_style: str = ""
    streamer_phrase_examples: List[str] = field(default_factory=list)
    prosody: Dict[str, Any] = field(default_factory=dict)
    web_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    long_term_memory: List[Dict[str, Any]] = field(default_factory=list)
    vts_token: str = ""


def _safe_json_load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _safe_text_load(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_knowledge_bundle() -> KnowledgeBundle:
    """Load the existing Charlatan knowledge assets into a single in-memory bundle."""
    bundle = KnowledgeBundle()

    bundle.style_audio_sources = _safe_json_load(KNOWLEDGE_DIR / "style_audio_sources.json", []) or []
    bundle.youtube_knowledge = _safe_json_load(KNOWLEDGE_DIR / "youtube_knowledge.json", []) or []
    bundle.web_knowledge = _safe_json_load(KNOWLEDGE_DIR / "web_learned_knowledge.json", []) or []
    bundle.chat_history = _safe_json_load(KNOWLEDGE_DIR / "chat_history.json", []) or []
    bundle.long_term_memory = _safe_json_load(KNOWLEDGE_DIR / "long_term_memory.json", []) or []

    bundle.streamer_style = _safe_json_load(KNOWLEDGE_DIR / "streamer_style.json", "") or ""
    bundle.phrasing_style = _safe_json_load(KNOWLEDGE_DIR / "phrasing_style.json", "") or ""
    bundle.streamer_phrase_examples = _safe_json_load(KNOWLEDGE_DIR / "streamer_phrase_examples.json", []) or []
    bundle.prosody = _safe_json_load(KNOWLEDGE_DIR / "streamer_prosody.json", {}) or {}

    vts_token_path = KNOWLEDGE_DIR / "vtube_studio_token.txt"
    if vts_token_path.exists():
        bundle.vts_token = _safe_text_load(vts_token_path).strip()

    return bundle


def ensure_style_audio_dir() -> Path:
    STYLE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    return STYLE_AUDIO_DIR


def list_style_audio_files() -> List[Path]:
    ensure_style_audio_dir()
    return sorted(
        [path for path in STYLE_AUDIO_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".mp3"],
        key=lambda item: item.name.casefold(),
    )


def list_style_audio_metadata() -> List[Dict[str, Any]]:
    bundle = load_knowledge_bundle()
    files = list_style_audio_files()
    metadata = bundle.style_audio_sources or []
    by_file = {
        str(item.get("file", "")).lower(): item
        for item in metadata
        if isinstance(item, dict) and isinstance(item.get("file"), str)
    }

    result: List[Dict[str, Any]] = []
    for path in files:
        row: Dict[str, Any] = {"file": str(path), "name": path.name, "exists": True}
        src = by_file.get(str(path).lower())
        if src is not None:
            row.update(src)
        result.append(row)
    return result


def load_runtime_state() -> Dict[str, Any]:
    bundle = load_knowledge_bundle()
    return {
        "repo_root": str(bundle.repo_root),
        "knowledge_dir": str(bundle.knowledge_dir),
        "style_audio_dir": str(bundle.style_audio_dir),
        "style_audio_files": [str(path) for path in list_style_audio_files()],
        "style_audio_meta": list_style_audio_metadata(),
        "youtube_knowledge_count": len(bundle.youtube_knowledge),
        "web_knowledge_count": len(bundle.web_knowledge),
        "chat_history_count": len(bundle.chat_history),
        "memory_count": len(bundle.long_term_memory),
        "streamer_style_length": len(bundle.streamer_style or ""),
        "phrasing_style_length": len(bundle.phrasing_style or ""),
        "prosody": bundle.prosody,
        "vts_token_loaded": bool(bundle.vts_token),
    }


__all__ = [
    "KnowledgeBundle",
    "KNOWLEDGE_DIR",
    "STYLE_AUDIO_DIR",
    "ensure_style_audio_dir",
    "list_style_audio_files",
    "list_style_audio_metadata",
    "load_knowledge_bundle",
    "load_runtime_state",
]
