"""
modules/smart_insights_core.py
-------------------------------
Smart Insights Core Engine:
1. Feature Engineering - Calculate budget features
2. Signal Detection - Detect spending patterns (including per-category)
3. Smart Insights - Generate human-readable insights
4. Real-time Updates - Trigger on expense add
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

# ═══════════════════════════════════════════════════════════════════════════════
# ── CONSTANTS & THRESHOLDS ────────────────────────────────────────────────────��─
# ═══════════════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    "burn_rate_warning": 1.1,
    "burn_rate_critical": 1.2,
    "burn_rate_good": 0.8,
    "saving_excellent": 0.20,
    "saving_good": 0.10,
    "saving_low": 0.05,
    "fixed_deviation_threshold": 0.05,
}

# Category mappings - move emergency from fixed to dynamic optional
FIXED_CATEGORIES = {
    "rent", "transport", "electricity", "water", 
    "gas", "internet", "mobile", "medical", "education"
}

# Emergency is treated as optional - counts against emergency budget plan
OPTIONAL_CATEGORIES = {"coffee", "shopping", "entertainment", "other"}  # Emergency has its own budget

# Arabic names for categories
CATEGORY_NAMES_AR = {
    "rent": "الإيجار",
    "transport": "المواصلات",
    "electricity": "الكهرباء",
    "water": "المياه",
    "gas": "الغاز",
    "internet": "الإنترنت",
    "mobile": "التليفون",
    "medical": "طبي",
    "education": "التعليم",
    "coffee": "القهوة",
    "shopping": "التسوق",
    "entertainment": "الترفيه",
    "other": "أخرى",
    "food": "الغذاء",
    "optional": "الاستمتاع",
    "saving": "الادخار",
    "emergency": "الطارئ",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ── DATA CLASSES ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureSet:
    """All computed features from user data."""
    food_spent: float
    optional_spent: float
    emergency_spent: float
    food_ratio: float
    optional_ratio: float
    emergency_ratio: float
    total_spent: float
    fixed_total: float
    dynamic_total: float
    saving: float
    saving_ratio: float
    income: float
    day: int
    food_plan: float
    optional_plan: float
    emergency_plan: float
    saving_plan: float
    burn_rate_food: float
    burn_rate_optional: float
    fixed_deviation: float
    days_left: int
    optional_overspend: float
    saving_affected: float
    fixed_details: dict = field(default_factory=dict)
    optional_details: dict = field(default_factory=dict)
    emergency_details: dict = field(default_factory=dict)


@dataclass
class Signal:
    """A detected signal from spending patterns."""
    name: str
    severity: Literal["info", "warning", "critical", "positive"]
    value: float | None = None
    category: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 1: FEATURE ENGINEERING ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features(
    plan: dict[str, float],
    fixed_expenses: dict[str, float],
    spending: dict[str, float],
    income: float,
    day: int,
) -> FeatureSet:
    """
    Compute all features from user data.
    Now includes per-category details for fixed and optional expenses.
    """
    food_plan = plan.get("food", 0)
    optional_plan = plan.get("optional", plan.get("enjoyment", 0))
    saving_plan = plan.get("saving", 0)
    emergency_plan = plan.get("emergency", 0)
    
    # Normalize keys to lowercase for consistent lookup
    spending_lower = {k.lower(): v for k, v in spending.items()}
    fixed_expenses_lower = {k.lower(): v for k, v in fixed_expenses.items()}
    
    # Calculate dynamic spending
    food_spent = spending_lower.get("food", 0)
    optional_spent = sum(spending_lower.get(cat, 0) for cat in OPTIONAL_CATEGORIES)
    emergency_spent = spending_lower.get("emergency", 0)
    
    # Calculate totals
    fixed_total = sum(fixed_expenses_lower.values())
    dynamic_total = sum(spending_lower.get(cat, 0) for cat in OPTIONAL_CATEGORIES)
    total_spent = sum(spending_lower.values())
    
    # Calculate fixed expense planned (reserved) and actual
    fixed_planned = sum(fixed_expenses_lower.values())  # Reserved amount
    actual_fixed = sum(spending_lower.get(cat, 0) for cat in FIXED_CATEGORIES)  # Actual paid
    
    # Calculate optional planned and actual
    optional_plan_amt = optional_plan
    
    # Emergency has its own budget category
    emergency_plan_amt = emergency_plan
    
    # Calculate saving - protected, only reduced by overspending
    # 1. Saving is reserved (saving_plan)
    # 2. If fixed overspends, it eats into saving
    # 3. If optional overspends, it eats into saving
    # 4. If emergency overspends, it eats into saving
    
    fixed_overspend = max(0, actual_fixed - fixed_planned)
    optional_overspend_val = max(0, optional_spent - optional_plan_amt)
    emergency_overspend_val = max(0, emergency_spent - emergency_plan_amt)
    
    # Apply overspends to saving (but saving can't go negative)
    saving = max(0, saving_plan - fixed_overspend - optional_overspend_val - emergency_overspend_val)
    
    # Calculate ratios - protect against zero income
    income_safe = income if income > 0 else 1.0  # Avoid division by zero
    food_ratio = (food_spent / food_plan) if food_plan > 0 else 0
    optional_ratio = (optional_spent / optional_plan) if optional_plan > 0 else 0
    emergency_ratio = (emergency_spent / emergency_plan) if emergency_plan > 0 else 0
    saving_ratio = (saving / income_safe)
    
    # Calculate burn rates
    day_factor = day / 30
    expected_food = day_factor * food_plan if food_plan > 0 else 0
    expected_optional = day_factor * optional_plan if optional_plan > 0 else 0
    
    burn_rate_food = (food_spent / expected_food) if expected_food > 0 else 0
    burn_rate_optional = (optional_spent / expected_optional) if expected_optional > 0 else 0
    
    # Calculate fixed deviation (already computed above)
    fixed_deviation = ((actual_fixed - fixed_planned) / fixed_planned) if fixed_planned > 0 else 0
    
    # Calculate optional overspend impact on saving
    optional_overspend = max(0, optional_spent - optional_plan)
    saving_affected = saving_plan - optional_overspend
    
    # Calculate per-category fixed details
    fixed_details = {}
    for cat in FIXED_CATEGORIES:
        planned = fixed_expenses_lower.get(cat, 0)
        actual = spending_lower.get(cat, 0)
        if planned > 0 or actual > 0:
            fixed_details[cat] = {
                "planned": planned,
                "actual": actual,
                "diff": actual - planned,
                "diff_pct": ((actual - planned) / planned * 100) if planned > 0 else 0,
                "status": "over" if actual > planned else "ok",
            }
    
    # Calculate per-category optional details
    optional_details = {}
    for cat in OPTIONAL_CATEGORIES:
        actual = spending_lower.get(cat, 0)
        if actual > 0:
            optional_details[cat] = {
                "actual": actual,
                "pct_of_optional": (actual / optional_spent * 100) if optional_spent > 0 else 0,
            }
    
    # Emergency has its own details
    emergency_details = {"actual": emergency_spent, "planned": emergency_plan}
    
    return FeatureSet(
        food_spent=food_spent,
        optional_spent=optional_spent,
        emergency_spent=emergency_spent,
        food_ratio=food_ratio,
        optional_ratio=optional_ratio,
        emergency_ratio=emergency_ratio,
        total_spent=total_spent,
        fixed_total=fixed_total,
        dynamic_total=dynamic_total,
        saving=saving,
        saving_ratio=saving_ratio,
        income=income,
        day=day,
        food_plan=food_plan,
        optional_plan=optional_plan,
        emergency_plan=emergency_plan,
        saving_plan=saving_plan,
        burn_rate_food=burn_rate_food,
        burn_rate_optional=burn_rate_optional,
        fixed_deviation=fixed_deviation,
        days_left=30 - day,
        optional_overspend=optional_overspend,
        saving_affected=saving_affected,
        fixed_details=fixed_details,
        optional_details=optional_details,
        emergency_details=emergency_details,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 2: SIGNAL DETECTION ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def detect_signals(features: FeatureSet) -> list[Signal]:
    """
    Detect signals from computed features.
    Includes per-category fixed and optional signals.
    """
    signals: list[Signal] = []
    t = THRESHOLDS
    
    # Food signals
    if features.burn_rate_food > t["burn_rate_critical"]:
        signals.append(Signal("fast_food_spending", "critical", features.burn_rate_food * 100))
    elif features.burn_rate_food > t["burn_rate_warning"]:
        signals.append(Signal("fast_food_spending", "warning", features.burn_rate_food * 100))
    
    if features.burn_rate_food < t["burn_rate_good"] and features.food_spent > 0:
        signals.append(Signal("slow_food_spending", "positive", features.burn_rate_food * 100))
    
    # Optional signals (aggregate)
    if features.burn_rate_optional > t["burn_rate_critical"]:
        signals.append(Signal("fast_optional_spending", "critical", features.burn_rate_optional * 100))
    elif features.burn_rate_optional > t["burn_rate_warning"]:
        signals.append(Signal("fast_optional_spending", "warning", features.burn_rate_optional * 100))
    
    if features.burn_rate_optional < t["burn_rate_good"] and features.optional_spent > 0:
        signals.append(Signal("slow_optional_spending", "positive", features.burn_rate_optional * 100))
    
    # Optional overspend signals
    if features.optional_overspend > 0:
        signals.append(Signal("optional_overspend", "warning", features.optional_overspend))
        signals.append(Signal("saving_affected", "info", features.saving_affected))
    
    # ── FIXED EXPENSES: Per-category signals ─────────────────────────────────
    for cat, detail in features.fixed_details.items():
        cat_name = CATEGORY_NAMES_AR.get(cat, cat)
        
        if detail["status"] == "over":
            signals.append(Signal(
                f"fixed_{cat}_over",
                "warning",
                detail["diff"],
                category=cat
            ))
        elif detail["actual"] > 0 and detail["planned"] > 0:
            signals.append(Signal(
                f"fixed_{cat}_ok",
                "positive",
                None,
                category=cat
            ))
    
    # ── OPTIONAL: Per-category signals (Coffee, Shopping, Entertainment, Other) ──
    for cat, detail in features.optional_details.items():
        cat_name = CATEGORY_NAMES_AR.get(cat, cat)
        
        # Only trigger if category has significant spending relative to its share
        if detail["actual"] > features.optional_plan * 0.2:
            signals.append(Signal(
                f"optional_{cat}_high",
                "warning",
                detail["actual"],
                category=cat
            ))
    
    # Emergency signals (separate from optional)
    if features.emergency_spent > 0:
        if features.emergency_ratio > 1.0:
            signals.append(Signal(
                "emergency_overspend",
                "critical",
                features.emergency_spent,
                category="emergency"
            ))
        elif features.emergency_ratio > 0.7:
            signals.append(Signal(
                "emergency_high",
                "warning",
                features.emergency_spent,
                category="emergency"
            ))
    
    # Saving signals
    if features.saving_ratio >= t["saving_excellent"]:
        signals.append(Signal("strong_saving", "positive", features.saving_ratio * 100))
    elif features.saving_ratio >= t["saving_good"]:
        signals.append(Signal("good_saving", "positive", features.saving_ratio * 100))
    elif features.saving_ratio < t["saving_low"] and features.income > 0:
        signals.append(Signal("low_saving", "warning", features.saving_ratio * 100))
    
    # Balanced behavior
    if (0.8 <= features.food_ratio <= 1.1 and 
        0.8 <= features.optional_ratio <= 1.1 and 
        features.total_spent > 0):
        signals.append(Signal("balanced_behavior", "positive", None))
    
    # End of month signals
    if features.day >= 25:
        if features.saving_affected >= features.saving_plan:
            signals.append(Signal("saving_at_risk", "critical", None))
        else:
            signals.append(Signal("month_winding_down", "info", None))
    
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 3: SMART INSIGHTS GENERATOR ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _get_phase(day: int) -> Literal["early", "mid", "late"]:
    if day <= 10:
        return "early"
    elif day <= 20:
        return "mid"
    return "late"


def _fmt(amount: float) -> str:
    return f"{amount:,.0f} ج.م"


INSIGHT_TEMPLATES: dict[str, list[str]] = {
    # Food
    "fast_food_spending": [
        "⚠️ إنفاق الغذاء يتسارع! وتيرة الإنفاق {value:.0f}% من المتوقع",
        "🔴 إنفاق الغذاء أعلى من المخطط بنسبة {value:.0f}%",
        "🍽️ تنبيه! معدل إنفاق الغذاء يتجاوز الحدود الآمنة",
    ],
    "slow_food_spending": [
        "💚 إنفاق الغذاء تحت السيطرة! توفير جيد",
        "👏 أداء ممتاز في التحكم بإنفاق الغذاء",
        "✅ وفّرت في الغذاء! راجع ادخار",
    ],
    
    # Optional (Aggregate)
    "fast_optional_spending": [
        "⚠️ مصاريف الاستمتاع تتسارع! نسبة الإنفاق {value:.0f}%",
        "🛍️ تجاوز في الاستمتاع! راجع تسوقك وقهوتك",
        "🎮 إنفاق الاستمتاع أعلى من المتوقع بـ {value:.0f}%",
    ],
    "slow_optional_spending": [
        "🌟 تحكم ممتاز في مصاريف الاستمتاع",
        "💪 أنت تحت السيطرة في الاستمتاع",
        "✅ وفّرت على الاستمتاع! ممتاز",
    ],
    "optional_overspend": [
        "🔴 تجاوزت ميزانية الاستمتاع بـ {value:,.0f} ج.م",
        "⚠️ الصرف الزائد على الاستمتاع: {value:,.0f} ج.م",
        "🚨 تجاوزت الاستمتاع بـ {value:,.0f} ج.م",
    ],
    "saving_affected": [
        "💸 هذا سيؤثر على ادخارك. المتبقي: {value:,.0f} ج.م",
        "⚠️ ادخارك سيتأثر بالمصاريف الزائدة",
        "📉 الادخار المتبقي: {value:,.0f} ج.م",
    ],
    
    # ── FIXED EXPENSES: Per-category templates ────────────────────────────────
    "fixed_rent_over": [
        "⚠️ الإيجار تجاوز الميزانية! الفرق {value:,.0f} ج.م",
        "🔴 إيجارك أعلى من المتوقع",
    ],
    "fixed_rent_ok": ["✅ إيجارك تحت السيطرة هذا الشهر"],
    
    "fixed_electricity_over": [
        "⚠️ الكهرباء تجاوزت المتوقع! الفرق {value:,.0f} ج.م",
        "🔴 فاتورة الكهرباء مرتفعة",
    ],
    "fixed_electricity_ok": ["✅ الكهرباء تحت السيطرة"],
    
    "fixed_water_over": [
        "⚠️ المياه تجاوزت الميزانية! الفرق {value:,.0f} ج.م",
    ],
    "fixed_water_ok": ["✅ المياه تحت السيطرة"],
    
    "fixed_gas_over": [
        "⚠️ الغاز تجاوز المتوقع! الفرق {value:,.0f} ج.م",
    ],
    "fixed_gas_ok": ["✅ الغاز تحت السيطرة"],
    
    "fixed_internet_over": [
        "⚠️ الإنترنت تجاوز الميزانية! الفرق {value:,.0f} ج.م",
    ],
    "fixed_internet_ok": ["✅ الإنترنت تحت السيطرة"],
    
    "fixed_mobile_over": [
        "⚠️ التليفون تجاوز المتوقع! الفرق {value:,.0f} ج.م",
    ],
    "fixed_mobile_ok": ["✅ التليفون تحت السيطرة"],
    
    "fixed_transport_over": [
        "⚠️ المواصلات تجاوزت الميزانية! الفرق {value:,.0f} ج.م",
        "🔴 مصاريف المواصلات أعلى من المتوقع",
    ],
    "fixed_transport_ok": ["✅ المواصلات تحت السيطرة"],
    
    "fixed_medical_over": [
        "⚠️ المصاريف الطبية تجاوزت المتوقع! الفرق {value:,.0f} ج.م",
    ],
    "fixed_medical_ok": ["✅ المصاريف الطبية تحت السيطرة"],
    
    "fixed_education_over": [
        "⚠️ التعليم تجاوز الميزانية! الفرق {value:,.0f} ج.م",
    ],
    "fixed_education_ok": ["✅ التعليم تحت السيطرة"],
    
    # ── OPTIONAL: Per-category templates ───────────────────────────────────
    "optional_coffee_high": [
        "☕ مصاريف القهوة مرتفعة! {value:,.0f} ج.م",
        "☕ القهوة تستهلك جزء كبير من الميزانية",
    ],
    "optional_shopping_high": [
        "🛒 التسوق مرتفع! {value:,.0f} ج.م",
        "🛒 مصاريف التسوق أعلى من المعتاد",
    ],
    "optional_entertainment_high": [
        "🎬 الترفيه مرتفع! {value:,.0f} ج.م",
        "🎬 إنفاق الترفيه يتزايد",
    ],
    "optional_other_high": [
        "📦 مصاريف أخرى مرتفعة! {value:,.0f} ج.م",
    ],
    
    # Saving
    "strong_saving": [
        "🎉 أداؤك في الادخار ممتاز! وفرت {value:.0f}% من دخلك",
        "💰 ادخار قوي! أنت توفر نسبة {value:.0f}% من دخلك",
        "🏆 ادخارك يتجاوز {value:.0f}%! أداء مالي رائع",
    ],
    "good_saving": [
        "💰 ادخارك جيد! وفرت {value:.0f}% من دخلك",
        "👍 نسبة الادخار {value:.0f}% جيدة",
    ],
    "low_saving": [
        "⚠️ نسبة الادخار منخفضة! فقط {value:.0f}% من دخلك",
        "🔴 ادخارك يحتاج تحسين! وفرت {value:.0f}% فقط",
    ],
    
    # Balance
    "balanced_behavior": [
        "⚖️ توازن جيد في إنفاقك!",
        "📊 ميزانيتك متوازنة بين الفئات",
        "✅ أداء متناسق في جميع الفئات",
    ],
    "saving_at_risk": [
        "🚨 تحذير! ادخارك في خطر مع نهاية الشهر",
        "🔴 الادخار مهدد هذا الشهر",
    ],
    "month_winding_down": [
        "📅 الشهر يوشك على الانتهاء. راجع إنفاقك للأيام المتبقية",
        "⏰ بقى {days} أيام. راجع خطة الإنفاق",
    ],
}


def generate_smart_insights(
    signals: list[Signal],
    features: FeatureSet,
    day: int,
) -> list[dict]:
    """
    Generate human-readable insights from signals and features.
    Now includes specific insights for each fixed and optional category.
    """
    insights = []
    phase = _get_phase(day)
    
    # Track used templates to avoid repetition
    used_signals = set()
    
    # Process signals in order of priority
    priority_order = ["critical", "warning", "info", "positive"]
    
    for severity in priority_order:
        severity_signals = [s for s in signals if s.severity == severity]
        
        for signal in severity_signals:
            if signal.name in used_signals:
                continue
            
            templates = INSIGHT_TEMPLATES.get(signal.name, [])
            if not templates:
                continue
            
            template = random.choice(templates)
            
            # Replace placeholders
            message = template
            # Format with .0f (percentage)
            if "{value:.0f}" in message:
                message = message.replace("{value:.0f}", f"{signal.value:.0f}")
            elif "{value}" in message:
                message = message.replace("{value}", f"{signal.value:.0f}")
            if "{value:,.0f}" in message:
                message = message.replace("{value:,.0f}", f"{signal.value:,.0f}")
            if "{days}" in message:
                message = message.replace("{days}", str(features.days_left))
            
            # Priority mapping
            priority_map = {
                "critical": "high",
                "warning": "medium",
                "info": "medium",
                "positive": "low",
            }
            
            insights.append({
                "message": message,
                "priority": priority_map.get(signal.severity, "medium"),
                "signal": signal.name,
                "severity": signal.severity,
                "category": signal.category,
            })
            
            used_signals.add(signal.name)
    
    # Phase-specific insights
    if phase == "early" and features.total_spent > 0:
        expected_pct = (day / 30) * 100
        actual_pct = (features.total_spent / (features.food_plan + features.optional_plan)) * 100 if features.food_plan + features.optional_plan > 0 else 0
        
        if actual_pct > expected_pct * 1.3:
            insights.insert(0, {
                "message": f"⚠️ إنفاق سريع جداً في بداية الشهر! استخدمت {actual_pct:.0f}% من الميزانية في أول {day} أيام",
                "priority": "high",
                "signal": "early_overspend",
                "severity": "warning",
                "category": None,
            })
    
    elif phase == "mid" and features.saving_ratio < 0.05:
        if features.optional_spent > features.optional_plan * 0.7:
            insights.append({
                "message": "🟡 نصيحة: راجع مصاريف الاستمتاع النصف الثاني من الشهر",
                "priority": "medium",
                "signal": "mid_month_tip",
                "severity": "info",
                "category": None,
            })
    
    elif phase == "late":
        remaining = features.saving_plan - features.saving
        if remaining > 0:
            daily_needed = remaining / features.days_left if features.days_left > 0 else 0
            insights.append({
                "message": f"📊 تحتاج {_fmt(daily_needed)} يومياً للوصل لهدف الادخار في {features.days_left} يوم",
                "priority": "medium",
                "signal": "end_month_plan",
                "severity": "info",
                "category": None,
            })
    
    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# ── STEP 4: REAL-TIME UPDATE ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def update_insights_on_expense(
    plan: dict[str, float],
    fixed_expenses: dict[str, float],
    spending: dict[str, float],
    income: float,
    day: int,
) -> list[dict]:
    """
    Full pipeline: compute features -> detect signals -> generate insights.
    Call this when user adds a new expense.
    """
    features = compute_features(plan, fixed_expenses, spending, income, day)
    signals = detect_signals(features)
    insights = generate_smart_insights(signals, features, day)
    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# ── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def get_spending_summary(features: FeatureSet) -> dict:
    """Get a summary dict of key spending metrics."""
    return {
        "total_spent": features.total_spent,
        "food_spent": features.food_spent,
        "optional_spent": features.optional_spent,
        "saving": features.saving,
        "saving_ratio": features.saving_ratio * 100,
        "food_burn_rate": features.burn_rate_food * 100,
        "optional_burn_rate": features.burn_rate_optional * 100,
    }


def get_category_breakdown(features: FeatureSet) -> dict:
    """Get breakdown of fixed and optional categories."""
    return {
        "fixed": features.fixed_details,
        "optional": features.optional_details,
    }