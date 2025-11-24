# -*- coding: utf-8 -*-
# vision.py
# Vision 모듈:
#   - CameraSource: OpenCV로 프레임 캡처
#   - FaceService: DeepFace 임베딩으로 사용자 등록/매칭(Firebase 저장/로드)
#   - HealthService: DeepFace emotion → 긍정/중립/부정 3분류 퍼센트 + mood/stress/fatigue/overall 단순 점수화
#       * primary_emotion은 3분류 대표값으로만 저장:
#           - 긍정 → "happy"
#           - 중립 → "sad"
#           - 부정 → "angry"
#   - DarkCircleService: MediaPipe FaceMesh로 눈 밑 ROI 추출 후 밝기 비교
#   - SkinToneService: 얼굴 ROI의 밝기(LAB L) 기반으로 단순 피부톤 점수(0~100)
#   - FatigueService: MediaPipe 눈 랜드마크(EAR) 기반 간단 피로도 점수(0~100)
#   - baseline은 baselines/{user_id}.json에 로컬 저장(조명 정규화 포함)

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

from models import UserProfile, FaceEmbedding, EmotionResult, HealthStatus
from firebase_client import FirebaseClient


class CameraSource:
    def __init__(self, device_index: int = 0):
        self.cap = cv2.VideoCapture(device_index)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap.isOpened():
            return False, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


class FaceService:
    def __init__(
            self,
            firebase: FirebaseClient,
            model_name: str = "Facenet",
            detector_backend: str = "opencv",
    ):
        self.firebase = firebase
        self.model_name = model_name
        self.detector_backend = detector_backend

    def _get_embedding(self, frame: np.ndarray) -> Optional[List[float]]:
        try:
            reps = DeepFace.represent(
                img_path=frame,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception:
            return None

        if not reps:
            return None

        rep = reps[0] if isinstance(reps, list) else reps
        emb = rep.get("embedding")
        if not emb:
            return None
        return [float(x) for x in emb]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or len(a) == 0:
            return -1.0
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0.0 or nb == 0.0:
            return -1.0
        return float(np.dot(va, vb) / (na * nb))

    def register_user(self, user: UserProfile, frame: np.ndarray) -> bool:
        emb = self._get_embedding(frame)
        if emb is None:
            return False
        embedding = FaceEmbedding(user_id=user.user_id, embedding=emb)
        self.firebase.save_face_embedding(embedding)
        return True

    def recognize_user(self, frame: np.ndarray) -> Optional[str]:
        query_emb = self._get_embedding(frame)
        if query_emb is None:
            return None

        candidates = self.firebase.load_all_face_embeddings()
        if not candidates:
            return None

        best_id: Optional[str] = None
        best_sim = -1.0
        for emb in candidates:
            sim = self._cosine_similarity(query_emb, emb.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = emb.user_id

        if best_id is None or best_sim < 0.7:
            return None
        return best_id


class HealthService:
    def __init__(self, detector_backend: str = "opencv"):
        self.detector_backend = detector_backend

    def _analyze_emotion(self, frame: np.ndarray) -> Optional[EmotionResult]:
        try:
            res = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception:
            return None

        data = res[0] if isinstance(res, list) else res
        primary = data.get("dominant_emotion") or data.get("dominant_emotion".upper())
        scores = data.get("emotion") or {}
        if not primary or not scores:
            return None

        scores_float = {k: float(v) for k, v in scores.items()}
        return EmotionResult(primary_emotion=str(primary), scores=scores_float, skin_tone=None)

    def _map_to_health(
            self,
            user_id: str,
            emotion: EmotionResult,
            skin_tone_score: float,
            fatigue_score: float,
    ) -> HealthStatus:
        scores = emotion.scores or {}

        happy = scores.get("happy", 0.0)
        surprise = scores.get("surprise", 0.0)
        sad = scores.get("sad", 0.0)
        fear = scores.get("fear", 0.0)
        angry = scores.get("angry", 0.0)
        disgust = scores.get("disgust", 0.0)

        # DeepFace emotion 스코어가 0~100인지 0~1인지 자동 보정
        maxv = max(scores.values()) if scores else 0.0
        scale = 100.0 if maxv > 1.5 else 1.0

        happy01 = happy / scale
        surprise01 = surprise / scale
        sad01 = sad / scale
        fear01 = fear / scale
        angry01 = angry / scale
        disgust01 = disgust / scale

        positive01 = happy01 + surprise01
        neutral01 = sad01 + fear01
        negative01 = angry01 + disgust01

        total01 = positive01 + neutral01 + negative01
        if total01 <= 1e-6:
            positive_pct = neutral_pct = negative_pct = 0.0
        else:
            positive_pct = positive01 / total01 * 100.0
            neutral_pct = neutral01 / total01 * 100.0
            negative_pct = negative01 / total01 * 100.0

        # primary_emotion을 3분류 대표로 교체
        if positive_pct >= neutral_pct and positive_pct >= negative_pct:
            emotion.primary_emotion = "happy"   # 긍정
        elif neutral_pct >= negative_pct:
            emotion.primary_emotion = "sad"     # 중립
        else:
            emotion.primary_emotion = "angry"   # 부정

        # 기존 건강 점수는 긍/부정 합으로 단순하게 계산 (0~100)
        mood_score = max(0.0, min(100.0, positive01 * 100.0))
        stress_score = max(0.0, min(100.0, (neutral01 + negative01) * 100.0))

        fatigue_score_final = fatigue_score if fatigue_score is not None else stress_score
        fatigue_score_final = max(0.0, min(100.0, float(fatigue_score_final)))

        overall_score = (
                                mood_score + (100.0 - stress_score) + (100.0 - fatigue_score_final)
                        ) / 3.0
        overall_score = max(0.0, min(100.0, overall_score))

        health = HealthStatus(
            user_id=user_id,
            mood_score=mood_score,
            stress_score=stress_score,
            fatigue_score=fatigue_score_final,
            overall_score=overall_score,
            skin_tone_score=max(0.0, min(100.0, float(skin_tone_score))),
            raw_emotion=emotion,
        )

        # DB에서 getattr로 읽을 수 있도록 3분류 퍼센트 부착
        health.positive_pct = float(positive_pct)
        health.neutral_pct = float(neutral_pct)
        health.negative_pct = float(negative_pct)

        return health

    def analyze_health(
            self,
            user_id: str,
            frame: np.ndarray,
            skin_tone_score: float = 0.0,
            fatigue_score: float = 0.0,
    ) -> Optional[HealthStatus]:
        emotion = self._analyze_emotion(frame)
        if emotion is None:
            return None

        emotion.skin_tone = float(skin_tone_score)

        return self._map_to_health(user_id, emotion, skin_tone_score, fatigue_score)


class DarkCircleService:
    def __init__(self, baseline_dir: str = "baselines"):
        self.baseline_dir = baseline_dir
        os.makedirs(self.baseline_dir, exist_ok=True)

        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _baseline_path(self, user_id: str) -> str:
        return os.path.join(self.baseline_dir, f"{user_id}.json")

    def _landmarks(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(rgb)
        if not res.multi_face_landmarks:
            return None
        return res.multi_face_landmarks[0].landmark

    def _eye_boxes(self, lms, w: int, h: int):
        left_idx = [33, 133, 159, 145, 160, 144]
        right_idx = [362, 263, 386, 374, 387, 373]

        def box(indices):
            xs = [lms[i].x * w for i in indices]
            ys = [lms[i].y * h for i in indices]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            return x1, y1, x2, y2

        return box(left_idx), box(right_idx)

    def _under_eye_roi(self, frame: np.ndarray):
        h, w, _ = frame.shape
        lms = self._landmarks(frame)
        if lms is None:
            return None

        (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2) = self._eye_boxes(lms, w, h)

        def under_box(x1, y1, x2, y2):
            eye_h = max(1, y2 - y1)
            uy1 = int(y2 + 0.15 * eye_h)
            uy2 = int(y2 + 1.00 * eye_h)
            ux1 = int(x1)
            ux2 = int(x2)

            uy1 = max(0, min(h - 1, uy1))
            uy2 = max(0, min(h, uy2))
            ux1 = max(0, min(w - 1, ux1))
            ux2 = max(0, min(w, ux2))

            if uy2 <= uy1 or ux2 <= ux1:
                return None
            return frame[uy1:uy2, ux1:ux2]

        left_roi = under_box(lx1, ly1, lx2, ly2)
        right_roi = under_box(rx1, ry1, rx2, ry2)

        if left_roi is None or right_roi is None:
            return None
        return left_roi, right_roi

    def _brightness_features(self, roi: np.ndarray):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        mean_v = float(v.mean())

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)
        mean_l = float(l.mean())

        return mean_v, mean_l

    def _compute_pair(self, frame: np.ndarray):
        rois = self._under_eye_roi(frame)
        if rois is None:
            return None

        left_roi, right_roi = rois
        lv, ll = self._brightness_features(left_roi)
        rv, rl = self._brightness_features(right_roi)
        mean_v = (lv + rv) / 2.0
        mean_l = (ll + rl) / 2.0

        face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_mean = float(face_gray.mean()) + 1e-6
        mean_v = mean_v / face_mean * 100.0
        mean_l = mean_l / face_mean * 100.0

        return mean_v, mean_l

    def save_baseline(self, user_id: str, frame: np.ndarray) -> bool:
        feats = self._compute_pair(frame)
        if feats is None:
            return False

        mean_v, mean_l = feats
        data = {"mean_v": mean_v, "mean_l": mean_l}

        with open(self._baseline_path(user_id), "w", encoding="utf-8") as f:
            json.dump(data, f)

        return True

    def load_baseline(self, user_id: str):
        path = self._baseline_path(user_id)
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data["mean_v"]), float(data["mean_l"])
        except Exception:
            return None

    def analyze_dark_circles(self, user_id: str, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        baseline = self.load_baseline(user_id)
        feats = self._compute_pair(frame)
        if feats is None:
            return None

        mean_v, mean_l = feats

        if baseline is None:
            score = 0.0
            status = "NO_BASELINE"
            diff = 0.0
            b_v, b_l = mean_v, mean_l
        else:
            b_v, b_l = baseline
            diff = ((b_v - mean_v) + (b_l - mean_l)) / 2.0
            score = max(0.0, min(100.0, diff * 2.0))
            status = "DARKER" if diff > 5.0 else "NORMAL"

        return {
            "dark_circle_score": score,
            "status": status,
            "diff": diff,
            "baseline_v": b_v,
            "baseline_l": b_l,
            "current_v": mean_v,
            "current_l": mean_l,
        }


class SkinToneService:
    def __init__(self, detector_backend: str = "opencv"):
        self.detector_backend = detector_backend

    def analyze_skin_tone(self, frame: np.ndarray) -> float:
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False,
            )
        except Exception:
            faces = None

        if not faces:
            return 0.0

        face_img = faces[0].get("face")
        if face_img is None:
            return 0.0

        face_img = np.asarray(face_img)

        if face_img.ndim == 3 and face_img.shape[-1] == 4:
            face_img = face_img[:, :, :3]

        if face_img.dtype != np.uint8:
            maxv = float(face_img.max()) if face_img.size > 0 else 1.0
            if maxv <= 1.5:
                face_img = face_img * 255.0
            face_img = np.clip(face_img, 0, 255).astype(np.uint8)

        if face_img.ndim == 3 and face_img.shape[-1] == 3:
            bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        elif face_img.ndim == 2:
            bgr = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
        else:
            return 0.0

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)
        l_mean = float(l.mean())

        score = (l_mean / 255.0) * 100.0
        return max(0.0, min(100.0, score))


class FatigueService:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze_fatigue(self, frame: np.ndarray) -> float:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(rgb)
        if not res.multi_face_landmarks:
            return 0.0

        lms = res.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left = {"p1": 33, "p2": 160, "p3": 158, "p4": 133, "p5": 153, "p6": 144}
        right = {"p1": 362, "p2": 385, "p3": 387, "p4": 263, "p5": 373, "p6": 380}

        def pt(i):
            return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float32)

        def ear(eye):
            p1, p2, p3, p4, p5, p6 = [pt(eye[k]) for k in ["p1", "p2", "p3", "p4", "p5", "p6"]]
            vert1 = np.linalg.norm(p2 - p6)
            vert2 = np.linalg.norm(p3 - p5)
            horiz = np.linalg.norm(p1 - p4) + 1e-6
            return float((vert1 + vert2) / (2.0 * horiz))

        avg_ear = (ear(left) + ear(right)) / 2.0

        ear_ref = 0.25
        fatigue = (ear_ref - avg_ear) / ear_ref * 100.0
        fatigue = max(0.0, min(100.0, fatigue))
        return float(fatigue)
