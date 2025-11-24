# -*- coding: utf-8 -*-
# models.py
# 프로젝트 공용 데이터 모델들.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any


@dataclass
class UserProfile:
    user_id: str
    name: str = "Unknown"


@dataclass
class FaceEmbedding:
    user_id: str
    embedding: List[float]
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EmotionResult:
    primary_emotion: str
    scores: Dict[str, float]
    skin_tone: Optional[float] = None



@dataclass
class HealthStatus:
    user_id: str
    mood_score: float
    stress_score: float
    fatigue_score: float
    overall_score: float
    skin_tone_score: float = 0.0
    raw_emotion: Optional[EmotionResult] = None

    # 추가 (기본값 있어서 호환성 OK)
    positive_pct: float = 0.0
    neutral_pct: float = 0.0
    negative_pct: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Recommendation:
    user_id: str
    health_status: HealthStatus
    supplements: List[str]
    foods: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VoiceCommand:
    user_id: Optional[str]
    raw_text: str
    intent: str
    slots: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
