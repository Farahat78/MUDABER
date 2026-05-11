"""
modules/analysis_engine.py
--------------------------
Time-Aware Spending Analysis Engine for the Smart Budget Reallocation System.

Core idea:
  At day D of a 30-day month, a perfectly linear spender would have consumed
  (D / 30) × planned_budget for each category.  We compare actual spending
  against this expected trajectory and emit a severity status.

Also contains:
  - Arabic category name registry (CATEGORY_ARABIC)
  - Statistical anomaly detection engine (z-score based)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Constants ──────────────────────────────────────────────────────────────────
DAYS_IN_MONTH:       int   = 30
HIGH_OVER_THRESHOLD: float = 1.20   # > 120% of expected → high_over
MIN_DAY:             int   = 1
MAX_DAY:             int   = 30

SpendingStatus = Literal["normal", "over", "high_over"]

# Anomaly detection configuration
MIN_TRANSACTIONS_FOR_ANOMALY: int   = 5
Z_SCORE_THRESHOLD:            float = 2.0
MIN_STD_FLOOR_RATIO:          float = 0.05


# ── Arabic Category Name Registry ─────────────────────────────────────────────
CATEGORY_ARABIC: dict[str, str] = {
    "food":          "الغذاء",
    "transport":     "المواصلات",
    "rent":          "الإيجار",
    "electricity":   "الكهرباء",
    "water":         "المياه",
    "gas":           "الغاز",
    "internet":      "الإنترنت",
    "mobile":        "التليفون",
    "shopping":      "التسوق",
    "education":     "التعليم",
    "medical":       "طبي",
    "entertainment": "الترفيه",
    "coffee":        "القهوة",
    "other":         "أخرى",
    "saving":        "الادخار",
    "emergency":     "طارئ",
    "enjoyment":    "استمتاع",
}


def get_category_ar(category: str) -> str:
    """Return the Arabic display name for a category, or the original if unmapped."""
    return CATEGORY_ARABIC.get(category.lower(), category)


# ── Data Containers ───────────────────────────────────────────────────────────
@dataclass
class CategoryAnalysis:
    """All computed metrics for a single budget category."""
    category:        str
    planned:         float
    spent:           float
    expected:        float
    overspend:       float
    pct_of_plan:     float
    pct_of_expected: float
    status:          SpendingStatus
    remaining:       float
    extra_fields:    dict = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """Full anomaly assessment for a single spending category."""
    category:          str
    is_anomaly:        bool
    has_enough_data:   bool
    transaction_count: int
    message:           str            # Arabic — ready for UI display
    z_score:           float | None = None
    mean:              float | None = None
    std:               float | None = None


# ── Core Time-Aware Analysis ──────────────────────────────────────────────────
def compute_expected(planned: float, day: int) -> float:
    """Linear pro-ration: expected spend by `day` of a 30-day month."""
    day = max(MIN_DAY, min(day, MAX_DAY))
    return (day / DAYS_IN_MONTH) * planned if planned > 0 else 0.0


def classify_status(spent: float, expected: float) -> SpendingStatus:
    """
    Severity classification:
      spent > expected × 1.20 → high_over
      spent > expected        → over
      otherwise               → normal
    """
    if expected == 0:
        return "over" if spent > 0 else "normal"
    ratio = spent / expected
    if ratio > HIGH_OVER_THRESHOLD:
        return "high_over"
    if ratio > 1.0:
        return "over"
    return "normal"


def analyse_category(
    category: str,
    planned:  float,
    spent:    float,
    day:      int,
) -> CategoryAnalysis:
    """Full analysis for a single budget category."""
    expected        = compute_expected(planned, day)
    status          = classify_status(spent, expected)
    overspend       = spent - expected
    remaining       = planned - spent
    pct_of_plan     = (spent / planned * 100) if planned > 0 else 0.0
    pct_of_expected = (spent / expected * 100) if expected > 0 else 0.0

    return CategoryAnalysis(
        category=category, planned=planned, spent=spent,
        expected=expected, overspend=overspend,
        pct_of_plan=pct_of_plan, pct_of_expected=pct_of_expected,
        status=status, remaining=remaining,
    )


OPTIONAL_SUBCATS = {"coffee", "shopping", "entertainment", "other", "emergency"}

def analyse_all_categories(
    plan:     dict[str, float],
    spending: dict[str, float],
    day:      int,
) -> dict[str, CategoryAnalysis]:
    """Run analyse_category() for every entry in the plan.
    Aggregates optional sub-categories into 'enjoyment'.
    Emergency has its own budget category in plan."""
    # Normalize spending keys to lowercase
    spending_lower = {k.lower(): v for k, v in spending.items()}
    
    # Calculate enjoyment total from sub-categories (NO emergency - it has its own budget)
    enjoyment_spent = sum(spending_lower.get(cat, 0) for cat in OPTIONAL_SUBCATS if cat != "emergency")
    
    # Emergency has its own category - get total emergency spending
    emergency_spent = spending_lower.get("emergency", 0)
    
    full_spending = dict(spending_lower)
    full_spending["enjoyment"] = enjoyment_spent
    full_spending["emergency"] = emergency_spent
    
    return {
        category: analyse_category(category, planned, full_spending.get(category.lower(), 0.0), day)
        for category, planned in plan.items()
    }


# ── Statistical Anomaly Detection ─────────────────────────────────────────────
def detect_category_anomaly(
    category: str,
    amounts:  list[float],
) -> AnomalyResult:
    """
    Z-score based anomaly detection on the latest transaction.

    Decision tree:
      count == 0         → silent (empty message)
      count == 1         → first entry, informational Arabic message, never flag
      1 < count < MIN    → collecting baseline, never flag
      count ≥ MIN        → z-score vs history; flag if z > Z_SCORE_THRESHOLD
    """
    count  = len(amounts)
    cat_ar = get_category_ar(category)

    if count == 0:
        return AnomalyResult(category=category, is_anomaly=False,
                             has_enough_data=False, transaction_count=0, message="")

    if count == 1:
        return AnomalyResult(
            category=category, is_anomaly=False,
            has_enough_data=False, transaction_count=1,
            message=(
                f"✔️ تم تسجيل أول عملية إنفاق في فئة {cat_ar}. "
                f"سيتم تحليل نمط الإنفاق تلقائياً بعد توفر بيانات كافية."
            ),
        )

    if count < MIN_TRANSACTIONS_FOR_ANOMALY:
        return AnomalyResult(
            category=category, is_anomaly=False,
            has_enough_data=False, transaction_count=count,
            message=(
                f"⏳ يتم جمع بيانات كافية لتحليل نمط الإنفاق في فئة {cat_ar} "
                f"({count} من {MIN_TRANSACTIONS_FOR_ANOMALY} معاملات مطلوبة)."
            ),
        )

    # Full z-score: history = all-but-last, candidate = latest
    history = amounts[:-1]
    latest  = amounts[-1]
    n       = len(history)
    mean    = sum(history) / n
    std     = (sum((x - mean) ** 2 for x in history) / n) ** 0.5
    std     = max(std, mean * MIN_STD_FLOOR_RATIO, 1.0)   # floor prevents zero-div
    z_score = (latest - mean) / std

    if z_score > Z_SCORE_THRESHOLD:
        return AnomalyResult(
            category=category, is_anomaly=True,
            has_enough_data=True, transaction_count=count,
            message=(
                f"⚠️ تم ملاحظة نمط إنفاق غير معتاد في فئة {cat_ar} — "
                f"المبلغ الأخير ({latest:,.0f} جنيه) أعلى بشكل ملحوظ من "
                f"متوسط إنفاقك المعتاد ({mean:,.0f} جنيه). "
                f"يُنصح بمراجعة هذا المصروف."
            ),
            z_score=z_score, mean=mean, std=std,
        )

    return AnomalyResult(
        category=category, is_anomaly=False,
        has_enough_data=True, transaction_count=count,
        message="", z_score=z_score, mean=mean, std=std,
    )


def detect_anomalies(
    plan:             dict[str, float],
    category_amounts: dict[str, list[float]],
) -> dict[str, AnomalyResult]:
    """Run per-category anomaly detection for every category in the plan."""
    return {
        category: detect_category_anomaly(category, category_amounts.get(category.lower(), []))
        for category in plan
    }
