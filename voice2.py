# -*- coding: utf-8 -*-
# voice2.py (발전 버전)
# 푸시투톡 (v키 누를 때만 인식)
# listen_sec 4초로 변경
# 기본 명령 추가

from __future__ import annotations

import json
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

import sounddevice as sd
from vosk import Model, KaldiRecognizer

from models import VoiceCommand
from firebase_client import FirebaseClient


class VoskVoiceRecognizer:
    def __init__(self, model_path: str, sample_rate: int = 16000, listen_sec: float = 4.0):
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

        print("말하는 중...")
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            end_time = datetime.utcnow().timestamp() + self.listen_sec
            while datetime.utcnow().timestamp() < end_time:
                if not self.q.empty():
                    data = self.q.get()
                    if rec.AcceptWaveform(data):
                        partial = rec.PartialResult()
                        text_partial = json.loads(partial).get("partial", "")
                        if text_partial:
                            print("부분 인식:", text_partial, end="\r")

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

        # 기본 명령
        if ("기준" in t and "저장" in t) or ("다크" in t and "등록" in t) or ("baseline" in t):
            intent = "SAVE_BASELINE"
        elif "건강" in t or "health" in t:
            intent = "SHOW_HEALTH"
        elif "오늘 상태" in t or "건강 상태" in t: # ex) 오늘 상태 어때
            intent = "SHOW_HEALTH"
        elif "다크서클" in t:
            intent = "SHOW_DARKCIRCLE"
        elif "추천" in t:
            intent = "SHOW_RECOMMENDATION"
        elif "불 켜" in t or "조명 켜" in t or "light on" in t:
            intent = "TURN_ON_LIGHT"
        elif "불 꺼" in t or "조명 꺼" in t or "light off" in t:
            intent = "TURN_OFF_LIGHT"

        # slot 예시: "조명 50%로 줄여줘"
        import re
        brightness_match = re.search(r"(\d{1,3})\s*%?", t)
        if brightness_match:
            slots["brightness"] = int(brightness_match.group(1))

        return VoiceCommand(
            user_id=user_id,
            raw_text=text,
            intent=intent,
            slots=slots,
        )


class VoiceService:
    def __init__(self, recognizer: VoskVoiceRecognizer, parser: SimpleCommandParser, firebase: Optional[FirebaseClient] = None):
        self.recognizer = recognizer
        self.parser = parser
        self.firebase = firebase

    def listen_for_command(self, user_id: Optional[str]) -> Optional[VoiceCommand]:
        text = self.recognizer.listen_and_recognize()
        if not text:
            print("음성 인식 실패 또는 입력 없음")
            return None

        cmd = self.parser.parse(text, user_id)

        # Firebase 로그 저장
        if self.firebase:
            try:
                self.firebase.push_voice_command({
                    "user_id": cmd.user_id,
                    "intent": cmd.intent,
                    "slots": cmd.slots,
                    "raw_text": cmd.raw_text,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                print("Firebase 기록 실패:", e)

        print("인식 결과:", cmd.intent, "| 텍스트:", cmd.raw_text, "| slots:", cmd.slots)
        return cmd
