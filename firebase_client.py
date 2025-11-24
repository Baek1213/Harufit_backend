# -*- coding: utf-8 -*-
# firebase_client.py
# Firestore + Realtime Database 쓰기를 이 파일로 몰아넣은 버전.
# - FaceEmbedding / HealthStatus / Recommendation / VoiceCommand / DarkCircles 모두 여기서 저장한다.
# - RTDB/Firestore에 들어가는 시간은 "YYYY-MM-DD HH:MM:SS" 까지만 저장.
# - 숫자 데이터는 소수점 2자리까지만 저장.
# - skin_tone은 하나의 값으로만 보낸다(별도 score 필드 없음).
# - overall_score는 DB에 저장하지 않는다(내부 계산용으로만 유지).
# - 대신 positive_pct / neutral_pct / negative_pct(긍정/중립/부정 %)를 저장한다.

from __future__ import annotations

from typing import List, Dict, Any

from models import FaceEmbedding, HealthStatus, Recommendation, VoiceCommand


class FirebaseClient:
    """firebase_admin 기반 Firestore + Realtime DB 클라이언트."""

    def __init__(self, cred_path: str, db_url: str, project_id: str):
        import firebase_admin
        from firebase_admin import credentials, firestore, db

        cred = credentials.Certificate(cred_path)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                cred,
                {"databaseURL": db_url, "projectId": project_id},
            )

        self.firestore_db = firestore.client()
        self.db = db

    # 얼굴 임베딩 ---------------------------

    def save_face_embedding(self, embedding: FaceEmbedding) -> None:
        doc_ref = self.firestore_db.collection("face_embeddings").document(embedding.user_id)
        doc_ref.set({
            "embedding": embedding.embedding,
            "user_id": embedding.user_id,
            "updated_at": embedding.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def load_all_face_embeddings(self) -> List[FaceEmbedding]:
        docs = self.firestore_db.collection("face_embeddings").stream()
        result: List[FaceEmbedding] = []
        for doc in docs:
            data = doc.to_dict()
            result.append(
                FaceEmbedding(
                    user_id=data["user_id"],
                    embedding=list(map(float, data["embedding"])),
                )
            )
        return result

    # 건강 상태 -----------------------------

    def push_health_status(self, health: HealthStatus) -> None:
        ts = health.created_at.strftime("%Y-%m-%d %H:%M:%S")

        # ✅ overall_score 제외, 3분류 퍼센트 포함
        rtdb_payload = {
            "mood_score": round(float(health.mood_score), 2),
            "stress_score": round(float(health.stress_score), 2),
            "fatigue_score": round(float(health.fatigue_score), 2),
            "primary_emotion": health.raw_emotion.primary_emotion if health.raw_emotion else None,
            "skin_tone": round(float(health.skin_tone_score), 2),

            "positive_pct": round(float(getattr(health, "positive_pct", 0.0)), 2),
            "neutral_pct": round(float(getattr(health, "neutral_pct", 0.0)), 2),
            "negative_pct": round(float(getattr(health, "negative_pct", 0.0)), 2),

            "created_at": ts,
        }

        # RTDB: 현재 상태
        self.db.reference(f"current_health/{health.user_id}").set(rtdb_payload)

        # Firestore: 히스토리
        self.firestore_db.collection("health_history").add({
            "user_id": health.user_id,
            **rtdb_payload,
        })

    # 다크서클 ------------------------------

    def push_dark_circles(self, user_id: str, dark: Dict[str, Any]) -> None:
        safe: Dict[str, Any] = {}
        for k, v in dark.items():
            if isinstance(v, (int, float)):
                safe[k] = round(float(v), 2)
            else:
                safe[k] = v
        self.db.reference(f"dark_circles/{user_id}").set(safe)

    # 추천 ----------------------------------

    def push_recommendation(self, reco: Recommendation) -> None:
        self.firestore_db.collection("recommendations").add({
            "user_id": reco.user_id,
            "supplements": reco.supplements,
            "foods": reco.foods,
            "created_at": reco.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # 음성 명령 -----------------------------

    def push_voice_command(self, cmd: VoiceCommand) -> None:
        ts = cmd.created_at.strftime("%Y-%m-%d %H:%M:%S")
        uid = cmd.user_id or "unknown"

        rtdb_payload = {
            "user_id": cmd.user_id,
            "raw_text": cmd.raw_text,
            "intent": cmd.intent,
            "slots": cmd.slots,
            "created_at": ts,
        }
        self.db.reference(f"voice_commands/{uid}").push(rtdb_payload)

        self.firestore_db.collection("voice_commands").add(rtdb_payload)
