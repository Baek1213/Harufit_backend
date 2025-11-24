# -*- coding: utf-8 -*-
# main.py
# 프로젝트 최상위 실행 파일.
# 구조:
#   - vision.py         : 카메라/얼굴 인식/감정 분석/다크서클/피부톤/피로도(DeepFace+MediaPipe+OpenCV)
#   - voice.py          : 오프라인 음성 인식 및 명령 파싱(Vosk)
#   - recommendation.py : 건강 상태/다크서클 기반 간단 추천 로직
#
# 동작:
#   1) 프레임 획득(OpenCV)
#   2) 사용자 매칭(DeepFace 임베딩)
#   3) 감정 기반 건강 점수 산출(DeepFace emotion)
#   4) 피부톤/피로도 산출(MediaPipe/OpenCV)
#   5) 다크서클 점수 산출(MediaPipe 눈 밑 ROI 밝기 비교)
#   6) RTDB/Firestore에 저장(FirebaseClient)
#   7) 음성 명령은 푸시투톡(v키)로만 인식, "기준 저장/다크서클 등록/baseline" → baseline 저장
#
# 키:
#   b : baseline 저장(현재 사용자 기준)
#   v : 2초 음성 인식(푸시투톡)
#   q : 종료

from __future__ import annotations

import time
from typing import Optional

import cv2

from models import UserProfile
from firebase_client import FirebaseClient
from vision import (
    CameraSource,
    FaceService,
    HealthService,
    DarkCircleService,
    SkinToneService,
    FatigueService,
)
from voice import VoskVoiceRecognizer, SimpleCommandParser, VoiceService
from recommendation import SimpleRecommendationEngine


class SmartMirrorBackend:
    def __init__(
        self,
        camera: CameraSource,
        face_service: FaceService,
        health_service: HealthService,
        dark_service: DarkCircleService,
        skin_service: SkinToneService,
        fatigue_service: FatigueService,
        voice_service: VoiceService,
        reco_engine: SimpleRecommendationEngine,
        firebase: FirebaseClient,
        default_user: UserProfile,
        analyze_interval_sec: float = 5.0,
    ):
        self.camera = camera
        self.face_service = face_service
        self.health_service = health_service
        self.dark_service = dark_service
        self.skin_service = skin_service
        self.fatigue_service = fatigue_service
        self.voice_service = voice_service
        self.reco_engine = reco_engine
        self.firebase = firebase
        self.default_user = default_user
        self.analyze_interval_sec = analyze_interval_sec
        self._last_analyze = 0.0

    def run(self) -> None:
        try:
            while True:
                ok, frame = self.camera.read_frame()
                if not ok or frame is None:
                    print("카메라 프레임 읽기 실패")
                    time.sleep(1.0)
                    continue

                user_id: Optional[str] = self.face_service.recognize_user(frame)
                if user_id is None:
                    user_id = self.default_user.user_id

                now = time.time()

                if now - self._last_analyze >= self.analyze_interval_sec:
                    self._last_analyze = now

                    skin_tone_score = self.skin_service.analyze_skin_tone(frame)
                    fatigue_score = self.fatigue_service.analyze_fatigue(frame)

                    health = self.health_service.analyze_health(
                        user_id, frame, skin_tone_score, fatigue_score
                    )
                    dark = self.dark_service.analyze_dark_circles(user_id, frame)

                    if health:
                        self.firebase.push_health_status(health)

                        reco = self.reco_engine.recommend(health=health, dark=dark)
                        self.firebase.push_recommendation(reco)

                    if dark is not None:
                        self.firebase.push_dark_circles(user_id, dark)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("b"):
                    saved = self.dark_service.save_baseline(user_id, frame)
                    print("다크서클 baseline 저장:", "성공" if saved else "실패")

                elif key == ord("v"):
                    print("음성 인식 시작(2초)... 말해봐")
                    cmd = self.voice_service.listen_for_command(user_id)
                    print("음성 인식 결과:", cmd.raw_text if cmd else None)

                    if cmd and cmd.intent == "SAVE_BASELINE":
                        saved = self.dark_service.save_baseline(user_id, frame)
                        print("음성으로 baseline 저장:", "성공" if saved else "실패")
                        self.firebase.push_voice_command(cmd)

                elif key == ord("q"):
                    break

                cv2.imshow("Smart Mirror Backend (Vision/Voice/Reco)", frame)

        finally:
            self.camera.release()
            cv2.destroyAllWindows()


def build_app() -> SmartMirrorBackend:
    firebase = FirebaseClient(
        cred_path="harufit-ce0bf-firebase-adminsdk-fbsvc-9bc112e263.json",
        db_url="https://harufit-ce0bf-default-rtdb.firebaseio.com",
        project_id="harufit-ce0bf",
    )

    camera = CameraSource(device_index=0)

    face_service = FaceService(firebase=firebase, model_name="Facenet", detector_backend="opencv")
    health_service = HealthService(detector_backend="opencv")
    dark_service = DarkCircleService(baseline_dir="baselines")

    skin_service = SkinToneService()
    fatigue_service = FatigueService()

    recognizer = VoskVoiceRecognizer(
        model_path="vosk-model-small-ko-0.22",
        sample_rate=16000,
        listen_sec=2.0,
    )
    parser = SimpleCommandParser()
    voice_service = VoiceService(recognizer, parser)

    reco_engine = SimpleRecommendationEngine()

    default_user = UserProfile(user_id="default", name="Guest")

    return SmartMirrorBackend(
        camera=camera,
        face_service=face_service,
        health_service=health_service,
        dark_service=dark_service,
        skin_service=skin_service,
        fatigue_service=fatigue_service,
        voice_service=voice_service,
        reco_engine=reco_engine,
        firebase=firebase,
        default_user=default_user,
    )


if __name__ == "__main__":
    app = build_app()
    app.run()