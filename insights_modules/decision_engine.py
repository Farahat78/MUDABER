"""
modules/decision_engine.py
--------------------------
Decision & Recommendation Engine - Smart Budget System.
Provides Arabic recommendations based on analysis results.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class Recommendation:
    """A single recommendation for a budget category."""
    message: str
    action: str  # "reduce", "maintain", "increase"
    reduction_amount: float = 0.0


def generate_all_recommendations(analyses: dict[str, Any]) -> dict[str, Recommendation]:
    """Generate recommendations for all categories based on their analysis."""
    recs = {}
    for category, analysis in analyses.items():
        recs[category] = generate_recommendation(analysis)
    return recs


def generate_recommendation(analysis: Any) -> Recommendation:
    """Generate a recommendation for a single category."""
    if analysis.pct_of_plan > 1.2:
        excess = analysis.spent - analysis.planned
        return Recommendation(
            message=f"تجاوزت الميزانية بنسبة {analysis.pct_of_plan:.0%}. "
                   f"فكر في تقليل {analysis.category} هذا الشهر.",
            action="reduce",
            reduction_amount=excess * 0.3
        )
    elif analysis.pct_of_plan < 0.5:
        return Recommendation(
            message=f"أنت وفير كبير في {analysis.category}. ممتاز!",
            action="maintain",
            reduction_amount=0.0
        )
    else:
        return Recommendation(
            message=f"إنفاقك في {analysis.category} مناسب.",
            action="maintain",
            reduction_amount=0.0
        )


def generate_weekly_summary(plan: dict[str, float], spending: dict[str, float]) -> str:
    """Generate an Arabic weekly summary string."""
    if not spending:
        return "لا توجد مصروفات هذا الأسبوع."
    
    total_spent = sum(spending.values())
    total_plan = sum(plan.values())
    
    if total_plan == 0:
        pct = 0
    else:
        pct = (total_spent / total_plan) * 100
    
    return f"مجموع المصروفات هذا الأسبوع: {total_spent:.0f}ج من الميزانية {total_plan:.0f}ج ({pct:.0f}%)"