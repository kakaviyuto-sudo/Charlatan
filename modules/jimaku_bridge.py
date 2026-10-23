"""字幕ちゃんの設定変換と認識結果リレー用スクリプト。

HTTPサーバーとAIのキュー処理はメイン側に残し、設定変換だけを分離する。
"""
from __future__ import annotations

import json
from pathlib import Path

JIMAKU_RECOGNITION_BRIDGE = r"""
<script id="neuro-ai-recognition-bridge">
(() => {
  if (window.__neuroAiRecognitionBridgeInstalled) return;
  window.__neuroAiRecognitionBridgeInstalled = true;
  const sendFinal = (event) => {
    try {
      if (!event || !event.results) return;
      const parts = [];
      const start = Number.isInteger(event.resultIndex) ? event.resultIndex : 0;
      for (let i = start; i < event.results.length; i++) {
        const result = event.results[i];
        if (result && result.isFinal && result[0] && result[0].transcript) {
          parts.push(String(result[0].transcript).trim());
        }
      }
      const text = parts.join('').trim();
      if (!text) return;
      fetch('/api/recognized', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text}),
        keepalive: true
      }).catch(() => {});
    } catch (_) {}
  };
  const wrap = (NativeCtor) => {
    if (!NativeCtor || NativeCtor.__neuroAiWrapped) return NativeCtor;
    function WrappedRecognition(...args) {
      const instance = new NativeCtor(...args);
      const seenEvents = new WeakSet();
      const relay = (event) => {
        if (event && typeof event === 'object' && !seenEvents.has(event)) {
          seenEvents.add(event);
          sendFinal(event);
        }
      };
      let resultHandler = null;
      try {
        Object.defineProperty(instance, 'onresult', {
          configurable: true,
          enumerable: true,
          get() { return resultHandler; },
          set(fn) {
            resultHandler = typeof fn === 'function' ? function(event) {
              relay(event);
              return fn.call(this, event);
            } : fn;
          }
        });
      } catch (_) {}
      const originalAdd = instance.addEventListener.bind(instance);
      instance.addEventListener = function(type, listener, options) {
        if (type === 'result' && typeof listener === 'function') {
          return originalAdd(type, function(event) {
            relay(event);
            return listener.call(this, event);
          }, options);
        }
        return originalAdd(type, listener, options);
      };
      return instance;
    }
    WrappedRecognition.prototype = NativeCtor.prototype;
    try { Object.setPrototypeOf(WrappedRecognition, NativeCtor); } catch (_) {}
    WrappedRecognition.__neuroAiWrapped = true;
    return WrappedRecognition;
  };
  if (window.SpeechRecognition) window.SpeechRecognition = wrap(window.SpeechRecognition);
  if (window.webkitSpeechRecognition) window.webkitSpeechRecognition = wrap(window.webkitSpeechRecognition);
})();
</script>

"""

def make_jimakuchan_settings_script(settings_path):
    """Exportされた字幕ちゃん設定を、同梱v1 index.htmlが読めるプリセット形式へ変換する。"""
    if not settings_path.is_file():
        return ""
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8-sig"))
        source = raw.get("data", raw)
        presets_in = source.get("presets", {}) if isinstance(source, dict) else {}
        if not isinstance(presets_in, dict) or not presets_in:
            print("字幕ちゃん設定ファイル形式を認識できません:", settings_path)
            return ""
        converted = {}
        for preset_id, preset in presets_in.items():
            if not isinstance(preset, dict):
                continue
            src = preset.get("settings", {})
            if not isinstance(src, dict):
                src = {}
            dst = {}
            direct = {
                "recog": "recog", "timer": "timer", "bouyomi": "bouyomi",
                "wordBoost": "word_boost_phrases",
                "wordBoostStrength": "word_boost_strength",
                "shortPause": "short_pause", "translationMethod": "translation_method",
                "recogModel": "recog_model", "textAlign": "textAlign",
                "whiteSpace": "whiteSpace", "bgcolor": "bgcolor",
                "gasKey": "gas_key",
            }
            for old, new in direct.items():
                if old in src:
                    dst[new] = src[old]
            if "vAlign" in src:
                dst["v_align"] = src["vAlign"]
            trans = src.get("trans")
            if isinstance(trans, list):
                for i, val in enumerate(trans[:3], start=1):
                    dst["trans" if i == 1 else f"trans{i}"] = val
            lines = src.get("lines")
            if isinstance(lines, list):
                font_keys = ["speech_text_font_selector", "trans_text_font_selector",
                             "trans_text2_font_selector", "trans_text3_font_selector"]
                for i, line in enumerate(lines[:4], start=1):
                    if not isinstance(line, dict):
                        continue
                    for field, suffix in (("size", "size"), ("weight", "weight"),
                                          ("color", "color"), ("strokeColor", "st_color"),
                                          ("strokeWidth", "st_width")):
                        if field in line:
                            dst[f"{suffix}{i}"] = str(line[field])
                    if "font" in line:
                        dst[font_keys[i-1]] = str(line["font"]).replace(" ", r"\ ")
            converted[preset_id] = {
                "name": str(preset.get("name", preset_id)),
                "settings": dst,
                "lastModified": preset.get("lastModified"),
            }
        if not converted:
            return ""
        current = source.get("current", "default")
        if current not in converted:
            current = "default" if "default" in converted else next(iter(converted))
        payload = json.dumps(converted, ensure_ascii=False).replace("</", "<\\/")
        current_js = json.dumps(str(current))
        return (
            '<script id="neuro-ai-jimakuchan-settings">\n'
            'try {\n'
            f' localStorage.setItem("jimakuChan_presets", {json.dumps(payload, ensure_ascii=False)});\n'
            f' localStorage.setItem("selectedPreset", {current_js});\n'
            '} catch(e) { console.warn("字幕ちゃん設定の自動適用に失敗:", e); }\n'
            '</script>\n'
        )
    except Exception as error:
        print("字幕ちゃん設定ファイルの読み込みエラー:", repr(error))
        return ""

