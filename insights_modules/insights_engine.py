"""
modules/insights_engine.py
--------------------------
Smart Insights Engine - Smart Budget System.
Stub implementation for backward compatibility.
"""

from __future__ import annotations
from typing import Any


def generate_smart_insights(
    plan: dict[str, float],
    actual_spending: dict[str, float],
    fixed_expenses: dict[str, float],
    day: int,
) -> list[dict[str, Any]]:
    """Generate smart insights - stub that returns minimal data."""
    total_plan = sum(plan.values())
    total_spent = sum(actual_spending.values())
    
    if total_plan == 0:
        pct = 0
    else:
        pct = (total_spent / total_plan) * 100
    
    insights = []
    
    if pct > 100:
        insights.append({
            "type": "warning",
            "message": "تجاوزت الميزانية الإجمالية!",
            "category": "general"
        })
    elif pct > 80:
        insights.append({
            "type": "info",
            "message": "اقتربت من الحد الأقصى.",
            "category": "general"
        })
    else:
        insights.append({
            "type": "positive",
            "message": "إنفاقك مناسب حتى الآن.",
            "category": "general"
        })
    
    return insights