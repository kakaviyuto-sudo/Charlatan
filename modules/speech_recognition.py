
"""マイク音声認識モジュール（単体テスト用）"""

import queue
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


class SpeechRecognizer:
    """マイク音声を取得し、日本語の音声認識を行う。"""

    def __init__(
        self,
        model_size="base",
        device="cpu",
        compute_type="int8",
        sample_rate=16000,
        threshold=0.010,
        silence_limit=0.9,
        max_utterance_seconds=15.0,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_limit = silence_limit
        self.max_utterance_seconds = max_utterance_seconds

        self._audio_queue = queue.Queue(maxsize=30)
        self._text_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._threads = []
        self.input_device = 3
        self._started = False
        self._model = None

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def load_model(self):
        """音声認識モデルを読み込む。"""
        if self._model is None:
            print(f"音声認識モデルを読み込み中: {self.model_size}")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            print("音声認識モデルの読み込み完了")

    def start(self):
        """マイク入力と音声認識を開始する。"""
        if self._started:
            print("音声認識はすでに起動しているよ")
            return

        self.load_model()
        self._stop_event.clear()

        self._threads = [
            threading.Thread(
                target=self._capture_worker,
                daemon=True,
                name="SpeechCapture",
            ),
            threading.Thread(
                target=self._recognition_worker,
                daemon=True,
                name="SpeechRecognition",
            ),
        ]

        for thread in self._threads:
            thread.start()

        self._started = True
        print("マイク音声認識を開始したよ")

    def _capture_worker(self):
        """マイク音声を取得し、無音区間で発話を区切る。"""
        block_seconds = 0.1
        block_size = int(self.sample_rate * block_seconds)
        preroll_blocks = 4
        preroll = []
        utterance = []
        speaking = False
        silence_seconds = 0.0
        utterance_seconds = 0.0

        try:
            with sd.InputStream(
                device=self.input_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
            ) as stream:
                while not self._stop_event.is_set():
                    audio, overflowed = stream.read(block_size)
                    if overflowed:
                        print("警告: マイク音声の一部を取りこぼしたよ")

                    chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
                    volume = float(np.sqrt(np.mean(chunk * chunk)))

                    if not speaking:
                        preroll.append(chunk.copy())
                        preroll = preroll[-preroll_blocks:]

                        if volume >= self.threshold:
                            speaking = True
                            utterance = [part.copy() for part in preroll]
                            utterance_seconds = len(utterance) * block_seconds
                            silence_seconds = 0.0
                    else:
                        utterance.append(chunk.copy())
                        utterance_seconds += block_seconds

                        if volume < self.threshold:
                            silence_seconds += block_seconds
                        else:
                            silence_seconds = 0.0

                        if (
                            silence_seconds >= self.silence_limit
                            or utterance_seconds >= self.max_utterance_seconds
                        ):
                            self._submit_audio(utterance)
                            utterance = []
                            speaking = False
                            silence_seconds = 0.0
                            utterance_seconds = 0.0
                            preroll = []

        except Exception as exc:
            if not self._stop_event.is_set():
                print(f"マイク取得エラー: {exc}")

    def _submit_audio(self, chunks):
        if not chunks:
            return

        audio = np.concatenate(chunks).astype(np.float32)

        if len(audio) < int(self.sample_rate * 0.2):
            return

        try:
            self._audio_queue.put_nowait(audio)
        except queue.Full:
            print("警告: 音声認識待ちキューが満杯なので発話をスキップしたよ")

    def _recognition_worker(self):
        """取得済み音声をWhisperで日本語に文字起こしする。"""
        while not self._stop_event.is_set():
            try:
                audio = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                segments, _ = self._model.transcribe(
                    audio,
                    language="ja",
                    beam_size=5,
                    best_of=3,
                    temperature=0,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                )

                text = "".join(segment.text for segment in segments).strip()
                if text:
                    print(f"認識結果: {text}")
                    try:
                        self._text_queue.put_nowait(text)
                    except queue.Full:
                        print("警告: 認識結果キューが満杯だよ")

            except Exception as exc:
                print(f"音声認識エラー: {exc}")
            finally:
                self._audio_queue.task_done()

    def get_text(self, timeout=0.15):
        """認識結果を取得する。結果がなければNoneを返す。"""
        try:
            return self._text_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """バックグラウンド処理を停止する。"""
        if not self._started:
            return

        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=2.0)

        self._threads = []
        self._started = False
        print("マイク音声認識を停止したよ")


if __name__ == "__main__":
    recognizer = SpeechRecognizer()

    try:
        recognizer.start()
        print("日本語で話してね。終了するには Ctrl+C を押してね。")

        while True:
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n終了処理中...")

    finally:
        recognizer.stop()
