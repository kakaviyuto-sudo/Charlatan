# Vision model for screen understanding
VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
VISION_MODEL_CANDIDATES = [
    VISION_MODEL,
    "qwen2.5vl:7b",
    "qwen2.5vl:3b",
    "qwen2.5-vl:7b",
    "qwen2.5-vl:3b",
    "gemma3:4b",
]
VISION_MODEL_CANDIDATES = list(dict.fromkeys(model for model in VISION_MODEL_CANDIDATES if model))


def call_vision_model(prompt, image_data):
    """Qwen2.5-VLなどの視覚モデルで画像とスクリーンテキストを理解する。"""
    last_error = None
    for model_name in VISION_MODEL_CANDIDATES:
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": prompt,
                        "images": [image_data],
                    }],
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "top_p": 0.8,
                        "repeat_penalty": 1.1,
                        "num_predict": 64,
                    },
                },
                timeout=60,
            )
            if response.ok:
                content = response.json().get("message", {}).get("content", "").strip()
                if content:
                    return content, model_name
            last_error = RuntimeError(
                f"Ollama視覚モデル応答失敗 (model={model_name}, HTTP {response.status_code}): "
                f"{response.text[:1200]}"
            )
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"Qwen2.5-VLまたは視覚モデルで画面理解に失敗しました: {last_error}")
    raise RuntimeError("Qwen2.5-VLまたは視覚モデルが利用できませんでした")


def generate_screen_comment():
    """
    画面上のWebページやアプリの内容を読み取り、
    テキスト、主要なオブジェクト、位置関係を把握し、
    自然な日本語で短い実況コメントを生成する。
    """
    global _last_screen_frame, _recent_screen_comments

    window_title, window_box = _get_foreground_window_info()

    with mss.MSS() as sct:
        if len(sct.monitors) < 2:
            print("画面認識: 利用できるモニターがありません。")
            return ""

        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

        if window_box:
            try:
                from PIL import ImageGrab
                active_image = ImageGrab.grab(bbox=window_box)
                if active_image.width >= 400 and active_image.height >= 250:
                    image = active_image.convert("RGB")
            except Exception:
                pass

        image.thumbnail((1024, 768))

        frame = image.convert("L").resize((160, 90))
        if _last_screen_frame is not None:
            diff = ImageChops.difference(frame, _last_screen_frame)
            mean_diff = ImageStat.Stat(diff).mean[0]
            if mean_diff < 3.0:
                return ""
        _last_screen_frame = frame.copy()

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70, optimize=True)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    recent_text = "\n".join(
        f"- {comment}" for comment in _recent_screen_comments[-8:]
    ) or "（まだありません）"

    prompt = f"""
あなたはユーザーの隣で同じ画面を見ている、気の利く日本語の相棒AIです。
画面の内容を注意深く見て、表示されているテキスト、主要なオブジェクト、UI部品、人物やアイコンの配置を把握し、
その場に合う自然な一言として日本語で返してください。

【重要ルール】
- 画面に表示されている文字や見出し、ボタン、入力欄、メニュー、画像、動画、アプリ画面を見て、どんな内容かを説明する
- オブジェクトの位置は「左上」「中央」「右下」などの相対位置で自然に書く。特に重要な要素は、どこにあるかを含めて説明する
- 文章のように機械的に羅列しない。見たことをそのまま言葉にして、感想や軽いツッコミを添える
- 画面の文字そのものを勝手に作らず、見えているものだけを基に説明する
- 画面に重要な情報がない場合は「SKIP」とだけ返す
- パスワード、個人情報、認証コードを読み上げない
- 1〜2文で短く、自然なタメ口にする
- 「ポイント」や見出し、箇条書きは禁止。実際に話す言葉だけを返す

【前面ウィンドウのタイトル】
{window_title or "取得できませんでした"}

【直近に話した画面コメント。繰り返さないこと】
{recent_text}
"""

    reply, model_name = call_vision_model(prompt, image_data)
    reply = reply.strip().strip('"「」 \n\t')
    reply = remove_point_wording(reply)

    if not reply or reply.upper() == "SKIP":
        return ""

    normalized = re.sub(r"\s+", "", reply).strip("。、！!？? ")
    for previous in _recent_screen_comments:
        previous_normalized = re.sub(r"\s+", "", previous).strip("。、！!？? ")
        if normalized and difflib.SequenceMatcher(None, normalized, previous_normalized).ratio() >= 0.72:
            return ""

    _recent_screen_comments.append(reply)
    _recent_screen_comments = _recent_screen_comments[-8:]
    print(f"Qwen2.5-VL画面認識（{model_name}）: {reply}")
    return reply


# 既存の generate_screen_comment を上書きする関数は、このまま使います。
# 上の関数定義が既に存在するため、下記の部分を置き換えてください。
# それ以外の処理は残し、モデル指定は Qwen2.5-VL を優先します。
