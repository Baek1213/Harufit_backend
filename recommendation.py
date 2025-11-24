# -*- coding: utf-8 -*-
# recommendation.py
# HealthStatus(+dark 결과)를 기반으로 간단 추천.

from __future__ import annotations

from typing import List, Optional, Dict, Any

from models import HealthStatus, Recommendation


class SimpleRecommendationEngine:
    def recommend(
            self,
            health: HealthStatus,
            dark: Optional[Dict[str, Any]] = None,
    ) -> Recommendation:
        supplements: List[str] = []
        foods: List[str] = []

        if health.overall_score < 50:
            supplements += ["종합비타민", "오메가-3"]
            foods += ["채소 위주의 식단", "연어, 고등어 등 지방이 풍부한 생선"]

        if health.stress_score > 60:
            supplements += ["마그네슘"]
            foods += ["허브티(캐모마일, 레몬밤)", "견과류"]

        if health.fatigue_score > 60:
            supplements += ["비타민 B 복합체"]
            foods += ["살코기", "계란", "콩류"]

        if dark and dark.get("dark_circle_score", 0) > 60:
            supplements += ["철분", "비타민 C"]
            foods += ["시금치, 브로콜리", "귤, 키위 등 비타민C 식품"]

        if not supplements and not foods:
            supplements = ["기본 종합비타민"]
            foods = ["균형 잡힌 일반 식단"]

        return Recommendation(
            user_id=health.user_id,
            health_status=health,
            supplements=supplements,
            foods=foods,
        )