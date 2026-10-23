
# Add topic duplication guard for proactive chat.
def _normalize_topic_text(text):
    text = str(text or "")
    text = text.lower()
    text = text.replace("〜", "~")
    text = re.sub(r"[\s\t\r\n\u3000]+", " ", text)
    text = re.sub(r"[「」『』\(\)\[\]{}\-–—_~]+", " ", text)
    text = re.sub(r"[。！？!?,,.:;]+", " ", text)
    text = re.sub(r"[\"'\\/]+", " ", text)
    return text.strip()


def _extract_topic_keywords(text):
    normalized = _normalize_topic_text(text)
    if not normalized:
        return []
    tokens = re.findall(r"[一-龥ぁ-んァ-ヶーA-Za-z0-9]{2,}", normalized)
    stop_words = {
        "こと", "これ", "それ", "あれ", "ない", "ある", "よう", "今日", "最近",
        "また", "今", "人", "話", "話題", "私", "あなた", "何", "どこ", "ここ",
        "時", "場合", "感じ", "思う", "そう", "かな", "ねえ", "ね", "ま", "み",
    }
    keywords = []
    for token in tokens:
        if len(token) >= 2 and token not in stop_words:
            keywords.append(token)
    return keywords


def _topic_similarity(left, right):
    a = _extract_topic_keywords(left)
    b = _extract_topic_keywords(right)
    if not a and not b:
        base_a = _normalize_topic_text(left)
        base_b = _normalize_topic_text(right)
        if not base_a or not base_b:
            return 0.0
        return difflib.SequenceMatcher(None, base_a, base_b).ratio()
    if not a or not b:
        return 0.0
    overlap = len(set(a) & set(b))
    union = len(set(a) | set(b))
    if union == 0:
        return 0.0
    overlap_ratio = overlap / union
    string_ratio = difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    return max(overlap_ratio, string_ratio * 0.7)


def _recent_topic_texts(limit=12):
    texts = []
    for item in history[-limit:]:
        content = item.get("content", "")
        if content:
            texts.append(content)
    return texts


def _filter_repeated_trends(trends, topic_limit=12, similarity_threshold=0.48):
    if not trends or not trends.strip():
        return ""

    recent_texts = _recent_topic_texts(limit=topic_limit)
    selected = []
    for line in trends.splitlines():
        item = line.strip()
        if not item:
            continue
        title = item.lstrip("- ").strip()
        if not title:
            continue
        if any(_topic_similarity(title, previous) >= similarity_threshold for previous in recent_texts):
            continue
        selected.append(item)

    if not selected:
        return ""
    return "\n".join(selected)


