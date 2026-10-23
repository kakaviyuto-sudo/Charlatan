# -*- coding: utf-8 -*-
"""knowledge 内の MP3 または TXT から話し方を学習する。既定では両方実行。"""
from pathlib import Path
import argparse
import importlib.util
import sys
import traceback

ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT / "main_nijisanji_vtube_studio_all_audio_learning_v14_sbv2.py",
    ROOT / "main_nijisanji_vtube_studio_all_audio_learning_v13_edge_tts_jimakuchan_settings_autoload_fixed_v2_modular.py",
    ROOT / "main_nijisanji_vtube_studio_all_audio_learning_v13_edge_tts_jimakuchan_settings_autoload_fixed_v2.py",
    ROOT / "qwen35_jimakuchan_integrated.py",
    ROOT / "main_nijisanji_vtube_studio_all_audio_learning_v13_mp3_style_learning.py",
]

def load_main(mode):
    required = {
        "mp3": ("learn_style_from_mp3",),
        "txt": ("learn_phrases_from_txt",),
        "both": ("learn_style_from_mp3", "learn_phrases_from_txt"),
    }[mode]
    errors = []
    for path in CANDIDATES:
        if not path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location("neuro_ai_learning_target", path)
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            if all(callable(getattr(mod, name, None)) for name in required):
                print(f"学習対象AI: {path.name}")
                return mod
            errors.append(f"{path.name}: 必要な学習関数がありません")
        except Exception as exc:
            errors.append(f"{path.name}: {exc!r}")
    detail = "\n".join(errors) if errors else "候補ファイルが見つかりませんでした"
    raise FileNotFoundError("対応するAI本体が見つからないか、必要な学習関数がありません。\n" + detail)

def main():
    parser = argparse.ArgumentParser(description="MP3/TXT 話し方学習")
    parser.add_argument("--mode", choices=("mp3", "txt", "both"), default="both",
                        help="実行する学習: mp3 / txt / both")
    args = parser.parse_args()

    knowledge = ROOT / "knowledge"
    knowledge.mkdir(exist_ok=True)
    mp3_dir = knowledge / "style_audio"
    mp3_dir.mkdir(exist_ok=True)

    txts = sorted(set(list(knowledge.glob("*.txt")) + list(ROOT.glob("*.txt"))))
    txts = [p for p in txts if p.name.lower() not in {
        "readme_導入手順.txt", "neuro_ai_統合起動手順.txt"
    }]
    mp3s = sorted(mp3_dir.glob("*.mp3"))

    print("=" * 58)
    print({"mp3": "MP3専用 話し方学習", "txt": "TXT専用 話し方学習",
           "both": "MP3・TXT 話し方自動学習"}[args.mode])
    print("MP3フォルダー:", mp3_dir)
    if args.mode in ("txt", "both"):
        print("TXT候補:", len(txts))
    if args.mode in ("mp3", "both"):
        print("MP3ファイル:", len(mp3s))
    print("=" * 58)

    if args.mode == "mp3" and not mp3s:
        print("[情報] MP3がありません。knowledge/style_audio にMP3を置いてください。")
        return
    if args.mode == "txt" and not txts:
        print("[情報] TXTがありません。knowledge またはこのスクリプトと同じフォルダーにTXTを置いてください。")
        return
    if args.mode == "both" and not mp3s and not txts:
        print("[情報] 学習対象のMP3/TXTがありません。")
        return

    mod = load_main(args.mode)

    if args.mode in ("txt", "both"):
        if txts:
            for path in txts:
                try:
                    print(f"\n[TXT] {path.name}")
                    print(mod.learn_phrases_from_txt(str(path)))
                except Exception as exc:
                    print(f"[警告] TXT学習失敗 {path.name}: {exc!r}")
        else:
            print("[情報] TXTがないため、TXT学習をスキップします。")

    if args.mode in ("mp3", "both"):
        if mp3s:
            try:
                print("\n[MP3] 既存AIのMP3学習処理を実行します。")
                print(mod.learn_style_from_mp3())
            except Exception as exc:
                print("[エラー] MP3学習に失敗:", repr(exc))
                traceback.print_exc()
                raise
        else:
            print("[情報] MP3がないため、MP3学習をスキップします。")

    print("\n学習処理が終わりました。保存先は knowledge 内の既存データです。")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[致命的エラー]", repr(exc))
        traceback.print_exc()
        input("\n終了するには Enter キーを押してください...")
        raise SystemExit(1)
    else:
        input("\n終了するには Enter キーを押してください...")
