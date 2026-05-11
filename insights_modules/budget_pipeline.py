"""
modules/budget_pipeline.py
--------------------------
Full Pipeline Orchestrator - Smart Budget System.
This is the ONLY module that app.py imports from.
"""

from __future__ import annotations

from typing import Any

from modules import database          as db
from modules import analysis_engine   as ae
from modules import decision_engine   as de
from modules import optimizer         as opt
from modules import insights_engine    as ie
from modules import achievements_engine as ach
from modules import smart_insights_core as sic
from modules import achievements_and_reports as ar


def check_user_status(
    user_id: str,
    plan:    dict[str, float],
    day:     int,
    month:   int = 0,
    year:    int = 0,
) -> dict[str, dict[str, Any]]:
    """Full pipeline: fetch -> analyse -> recommend.
    If month/year provided, get monthly spending only."""
    if month > 0 and year > 0:
        spending = db.get_monthly_spending(user_id, month, year)
    else:
        spending = db.get_all_categories_spent(user_id)
    
    normalised_plan = {k.lower(): float(v) for k, v in plan.items()}

    analyses        = ae.analyse_all_categories(normalised_plan, spending, day)
    recommendations = de.generate_all_recommendations(analyses)

    output: dict[str, dict[str, Any]] = {}
    for category in normalised_plan:
        analysis = analyses[category]
        rec      = recommendations.get(category)
        output[category] = {
            "spent":            analysis.spent,
            "expected":         analysis.expected,
            "planned":          analysis.planned,
            "remaining":        analysis.remaining,
            "overspend":        analysis.overspend,
            "pct_of_plan":      analysis.pct_of_plan,
            "pct_of_expected":  analysis.pct_of_expected,
            "status":           analysis.status,
            "recommendation":   rec.message          if rec else "لا توجد توصية.",
            "action":           rec.action           if rec else "maintain",
            "reduction_amount": rec.reduction_amount if rec else 0.0,
        }
    return output


def record_expense(
    user_id:  str,
    category: str,
    amount:   float,
    date:     str | None = None,
) -> int:
    """Add a new expense. Raises ValueError on invalid input."""
    return db.add_expense(user_id, category, amount, date)


def get_weekly_summary(user_id: str, plan: dict[str, float]) -> str:
    """Return an Arabic weekly spending summary string."""
    weekly_spending = db.get_weekly_summary(user_id)
    normalised_plan = {k.lower(): float(v) for k, v in plan.items()}
    return de.generate_weekly_summary(normalised_plan, weekly_spending)


def get_anomalies(
    user_id: str,
    plan:    dict[str, float],
) -> dict[str, ae.AnomalyResult]:
    """Per-category statistical anomaly detection (z-score based)."""
    normalised_plan   = {k.lower(): float(v) for k, v in plan.items()}
    category_amounts  = {cat: db.get_transaction_amounts(user_id, cat)
                         for cat in normalised_plan}
    return ae.detect_anomalies(normalised_plan, category_amounts)


def reset_user(user_id: str) -> int:
    """Delete all expenses for a user. Returns deleted row count."""
    return db.clear_user_expenses(user_id)


def serialize_status(status_dict: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert check_user_status output to a flat list."""
    result = []
    for cat, data in status_dict.items():
        item = {"category": cat}
        item.update(data)
        result.append(item)
    return result


def run_optimization(
    user_id:          str,
    plan:             dict[str, float],
    priority_weights: dict[str, float] | None = None,
    lifestyle:        str                      = "balanced",
    fixed_expenses:   dict[str, float] | None  = None,
    category_floors:  dict[str, float] | None  = None,
    category_caps:    dict[str, float] | None  = None,
    prev_allocations: dict[str, float] | None  = None,
) -> opt.OptimizationResult:
    """Run constrained optimization for a user's budget."""
    normalised_plan = {k.lower(): float(v) for k, v in plan.items()}
    actual_spending = db.get_all_categories_spent(user_id)

    optimizer = opt.build_optimizer(
        total_budget     = sum(normalised_plan.values()),
        plan             = normalised_plan,
        priority_weights = priority_weights,
        lifestyle        = lifestyle,
        fixed_expenses   = fixed_expenses,
        category_floors  = category_floors,
        category_caps    = category_caps,
    )
    return optimizer.optimize(
        prev_allocations=prev_allocations,
        actual_spending=actual_spending,
    )


def run_dynamic_reallocation(
    user_id:             str,
    plan:                dict[str, float],
    current_allocations: dict[str, float],
    priority_weights:    dict[str, float] | None = None,
    lifestyle:           str                      = "balanced",
    fixed_expenses:      dict[str, float] | None  = None,
    category_floors:     dict[str, float] | None  = None,
    category_caps:       dict[str, float] | None  = None,
) -> opt.OptimizationResult:
    """Dynamic re-optimization triggered by overspending."""
    normalised_plan    = {k.lower(): float(v) for k, v in plan.items()}
    normalised_current = {k.lower(): float(v) for k, v in current_allocations.items()}
    actual_spending    = db.get_all_categories_spent(user_id)

    optimizer = opt.build_optimizer(
        total_budget     = sum(normalised_plan.values()),
        plan             = normalised_plan,
        priority_weights = priority_weights,
        lifestyle        = lifestyle,
        fixed_expenses   = fixed_expenses,
        category_floors  = category_floors,
        category_caps    = category_caps,
    )
    return optimizer.reallocate_on_overspend(
        current_allocations=normalised_current,
        actual_spending=actual_spending,
    )


def optimization_result_to_display(
    result: opt.OptimizationResult,
    plan:   dict[str, float],
) -> list[dict[str, Any]]:
    """Flatten OptimizationResult for Streamlit display."""
    rows = []
    for cr in result.category_results:
        current   = float(plan.get(cr.category, 0))
        opt_alloc = cr.opt_allocation
        change    = ((opt_alloc - current) / current * 100) if current > 0 else 0.0
        rows.append({
            "category":     cr.category,
            "current_plan": current,
            "opt_alloc":    opt_alloc,
            "change_pct":   round(change, 1),
            "utility":      cr.utility_score,
            "explanation":  cr.explanation,
        })
    return rows


def get_smart_insights(
    user_id: str,
    plan: dict[str, float],
    day: int,
    fixed_expenses: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate smart insights comparing:
    - Fixed expenses (user input) vs actual spending
    - AI plan categories (saving, emergency, optional, food) vs actual
    - Time-aware velocity analysis
    """
    actual_spending = db.get_all_categories_spent(user_id)
    return ie.generate_smart_insights(plan, actual_spending, fixed_expenses or {}, day)


def evaluate_user_achievements(
    user_id: str,
    plan:    dict[str, float],
    day:     int,
) -> list[dict[str, Any]]:
    """Evaluate and award achievements based on spending behavior."""
    spending = db.get_all_categories_spent(user_id)
    return ach.evaluate_achievements(user_id, spending, plan, day)


def get_all_achievements(
    user_id: str,
) -> list[dict[str, Any]]:
    """Get all achievements (earned and unearned) for a user."""
    return ach.get_user_achievements(user_id)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Smart Insights Core API ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def get_smart_insights_core(
    user_id: str,
    plan: dict[str, float],
    day: int,
    fixed_expenses: dict[str, float] | None = None,
    income: float = 10000.0,
) -> list[dict[str, Any]]:
    """
    Generate smart insights using the core engine.
    Pipeline: compute features -> detect signals -> generate insights.
    """
    actual_spending = db.get_all_categories_spent(user_id)
    return sic.update_insights_on_expense(
        plan, fixed_expenses or {}, actual_spending, income, day
    )


def compute_budget_features(
    plan: dict[str, float],
    fixed_expenses: dict[str, float],
    spending: dict[str, float],
    income: float,
    day: int,
) -> dict:
    """Compute all budget features."""
    features = sic.compute_features(plan, fixed_expenses, spending, income, day)
    return {
        "food_spent": features.food_spent,
        "optional_spent": features.optional_spent,
        "food_ratio": features.food_ratio * 100,
        "optional_ratio": features.optional_ratio * 100,
        "total_spent": features.total_spent,
        "fixed_total": features.fixed_total,
        "saving": features.saving,
        "saving_ratio": features.saving_ratio * 100,
        "burn_rate_food": features.burn_rate_food * 100,
        "burn_rate_optional": features.burn_rate_optional * 100,
        "optional_overspend": features.optional_overspend,
        "saving_affected": features.saving_affected,
        "days_left": features.days_left,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── Achievements & Reports API ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def get_user_achievements(
    user_id: str,
    plan: dict[str, float],
    fixed_expenses: dict[str, float],
    income: float,
    day: int,
    history: list[dict] | None = None,
) -> dict:
    """
    Generate achievements for user based on current spending and history.
    """
    spending = db.get_all_categories_spent(user_id)
    features = sic.compute_features(plan, fixed_expenses, spending, income, day)
    signals = sic.detect_signals(features)
    
    result = ar.generate_achievements(features, signals, fixed_expenses, history)
    
    return {
        "new_achievements": result.new_achievements,
        "all_achievements": result.all_achievements,
    }


def get_monthly_report(
    user_id: str,
    plan: dict[str, float],
    fixed_expenses: dict[str, float],
    income: float,
    day: int,
) -> dict:
    """
    Generate monthly report for user.
    """
    spending = db.get_all_categories_spent(user_id)
    features = sic.compute_features(plan, fixed_expenses, spending, income, day)
    signals = sic.detect_signals(features)
    
    report = ar.generate_monthly_report(features, signals)
    
    return {
        "stats": report.stats,
        "narrative": report.narrative,
        "insights": report.insights,
        "warnings": report.warnings,
        "strengths": report.strengths,
    }