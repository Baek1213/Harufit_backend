# -*- coding: utf-8 -*-
# voice.py
# Vosk 기반 오프라인 STT + 간단 명령 파서.
# v키(푸시투톡)로 listen_sec 동안만 듣는다.

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

import sounddevice as sd
from vosk import Model, KaldiRecognizer

from models import VoiceCommand


class VoskVoiceRecognizer:
    def __init__(
            self,
            model_path: str,
            sample_rate: int = 16000,
            listen_sec: float = 2.0,
    ):
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.listen_sec = listen_sec
        self.q: "queue.Queue[bytes]" = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        if status:
            return
        self.q.put(bytes(indata))

    def listen_and_recognize(self) -> Optional[str]:
        rec = KaldiRecognizer(self.model, self.sample_rate)
        self.q.queue.clear()

        with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=self._callback,
        ):
            end_time = datetime.utcnow().timestamp() + self.listen_sec
            while datetime.utcnow().timestamp() < end_time:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    pass

        try:
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            return text if text else None
        except Exception:
            return None


class SimpleCommandParser:
    def parse(self, text: str, user_id: Optional[str]) -> VoiceCommand:
        t = text.lower()
        intent = "UNKNOWN"
        slots: Dict[str, Any] = {}

        if ("기준" in t and "저장" in t) or ("다크" in t and "등록" in t) or ("baseline" in t):
            intent = "SAVE_BASELINE"
        elif "건강" in t or "health" in t:
            intent = "SHOW_HEALTH"
        elif "불 켜" in t or "조명 켜" in t or "light on" in t:
            intent = "TURN_ON_LIGHT"
        elif "불 꺼" in t or "조명 꺼" in t or "light off" in t:
            intent = "TURN_OFF_LIGHT"

        return VoiceCommand(
            user_id=user_id,
            raw_text=text,
            intent=intent,
            slots=slots,
        )


class VoiceService:
    def __init__(self, recognizer: VoskVoiceRecognizer, parser: SimpleCommandParser):
        self.recognizer = recognizer
        self.parser = parser

    def listen_for_command(self, user_id: Optional[str]) -> Optional[VoiceCommand]:
        text = self.recognizer.listen_and_recognize()
        if not text:
            return None
        return self.parser.parse(text, user_id)
