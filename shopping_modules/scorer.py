"""
scorer.py
════════════════════════════════════════════════════════════════════════
Intent scoring system — measures how well the final list satisfies
the user's stated intent.

Score: 0–100 (higher = better match)
Breakdown into sub-scores for transparency in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from fuzzy_matcher import find_in_list, norm, CONFIDENT
from nlp_engine import Action


@dataclass
class ScoreBreakdown:
    total:            float   # 0–100 overall
    intent_coverage:  float   # Were all intents addressed?
    result_quality:   float   # Did the changes make sense?
    budget_adherence: float   # Is the list within budget?
    diversity:        float   # Are ≥ 3 categories still present?
    details:          list[str] = None   # Human-readable notes

    def __post_init__(self):
        if self.details is None:
            self.details = []

    @property
    def grade(self) -> str:
        if self.total >= 90: return "A ممتاز"
        if self.total >= 75: return "B جيد جداً"
        if self.total >= 60: return "C جيد"
        if self.total >= 45: return "D مقبول"
        return "F يحتاج مراجعة"

    @property
    def color(self) -> str:
        if self.total >= 75: return "#27ae60"
        if self.total >= 50: return "#f39c12"
        return "#e74c3c"


def score_result(
    original_list: pd.DataFrame,
    final_list: pd.DataFrame,
    actions: list[Action],
    budget: float,
) -> ScoreBreakdown:
    """
    Scores the final shopping list against the user's actions.

    Parameters
    ----------
    original_list : list BEFORE modifications
    final_list    : list AFTER modifications
    actions       : parsed Action objects from nlp_engine
    budget        : monthly budget in EGP

    Returns
    -------
    ScoreBreakdown with scores 0–100 and human-readable details
    """
    details: list[str] = []
    scores:  dict[str, float] = {}

    # ── 1. Intent Coverage (40 pts) ───────────────────────────────────────────
    # How many intents were successfully applied?
    if not actions:
        scores["intent_coverage"] = 70.0  # neutral if no actions parsed
        details.append("ℹ️ لا توجد أوامر محددة للتقييم")
    else:
        covered = 0
        for action in actions:
            covered += _check_action_applied(action, original_list, final_list)
        pct = covered / len(actions) if actions else 1.0
        scores["intent_coverage"] = round(pct * 100, 1)
        details.append(f"✅ نُفِّذ {covered}/{len(actions)} أمر")

    # ── 2. Result Quality (25 pts) ────────────────────────────────────────────
    # Is the final list reasonable? (not too short, not all same category)
    n_items = len(final_list)
    n_cats  = final_list["category"].nunique() if not final_list.empty else 0

    if n_items == 0:
        scores["result_quality"] = 0.0
        details.append("❌ القائمة فارغة!")
    elif n_items < 5:
        scores["result_quality"] = 40.0
        details.append(f"⚠️ القائمة قصيرة جداً ({n_items} منتجات)")
    elif n_cats < 3:
        scores["result_quality"] = 55.0
        details.append(f"⚠️ فئات قليلة ({n_cats})")
    else:
        score = min(100.0, 70.0 + n_cats * 4.0 + min(n_items, 20) * 0.5)
        scores["result_quality"] = round(score, 1)
        details.append(f"✅ قائمة متوازنة: {n_items} منتج, {n_cats} فئات")

    # ── 3. Budget Adherence (20 pts) ──────────────────────────────────────────
    if budget <= 0 or budget == float("inf"):
        scores["budget_adherence"] = 100.0
    else:
        total_cost = float(final_list["total_price"].sum()) if not final_list.empty else 0.0
        pct_used   = total_cost / budget

        if pct_used <= 1.0:
            # Under budget: ideal range 70–100%
            if pct_used >= 0.70:
                scores["budget_adherence"] = 100.0
                details.append(f"✅ الميزانية: {pct_used*100:.0f}% مستخدمة")
            else:
                scores["budget_adherence"] = round(60.0 + pct_used * 50, 1)
                details.append(f"⚠️ الميزانية: {pct_used*100:.0f}% فقط (يمكن إضافة المزيد)")
        else:
            # Over budget
            scores["budget_adherence"] = max(0.0, round(100 - (pct_used - 1) * 200, 1))
            details.append(f"❌ تجاوز الميزانية: {pct_used*100:.0f}%")

    # ── 4. Diversity (15 pts) ─────────────────────────────────────────────────
    if final_list.empty:
        scores["diversity"] = 0.0
    else:
        n_cats  = final_list["category"].nunique()
        # Ideal: ≥ 6 categories
        score   = min(100.0, (n_cats / 6.0) * 100.0)
        scores["diversity"] = round(score, 1)
        if n_cats >= 6:
            details.append(f"✅ تنوع ممتاز: {n_cats} فئات")
        elif n_cats >= 3:
            details.append(f"⚠️ تنوع مقبول: {n_cats} فئات")
        else:
            details.append(f"❌ تنوع ضعيف: {n_cats} فئات فقط")

    # ── Weighted total ────────────────────────────────────────────────────────
    weights = {
        "intent_coverage":  0.40,
        "result_quality":   0.25,
        "budget_adherence": 0.20,
        "diversity":        0.15,
    }
    total = sum(scores[k] * weights[k] for k in weights)

    return ScoreBreakdown(
        total            = round(total, 1),
        intent_coverage  = scores["intent_coverage"],
        result_quality   = scores["result_quality"],
        budget_adherence = scores["budget_adherence"],
        diversity        = scores["diversity"],
        details          = details,
    )


def _check_action_applied(
    action: Action,
    original: pd.DataFrame,
    final: pd.DataFrame,
) -> float:
    """
    Returns 0–1 float indicating how well this action was applied.
    Uses structural checks: did the right thing change?
    """
    t = action.type

    if t == "remove":
        query = action.target_ar or action.target
        orig_hits = find_in_list(query, original)
        fin_hits  = find_in_list(query, final)
        if not orig_hits:
            return 0.5   # wasn't in list anyway
        orig_names = {h.product_name for h in orig_hits if h.confident}
        fin_names  = {h.product_name for h in fin_hits  if h.confident}
        removed    = orig_names - fin_names
        return 1.0 if removed else 0.2

    if t == "increase":
        query = action.target_ar or action.target
        o = find_in_list(query, original)
        f = find_in_list(query, final)
        if not o or not f:
            return 0.5
        o_qty = original[original["product_name"] == o[0].product_name]["quantity"].sum()
        f_qty = final[final["product_name"]   == f[0].product_name]["quantity"].sum()
        if f_qty > o_qty:
            return 1.0
        # Maybe category-level: check total category qty
        if action.scope == "category":
            cat = action.category
            o_cat_qty = original[original["category"] == cat]["quantity"].sum()
            f_cat_qty = final[final["category"]   == cat]["quantity"].sum()
            return 1.0 if f_cat_qty >= o_cat_qty else 0.3
        return 0.3

    if t == "decrease":
        query = action.target_ar or action.target
        o = find_in_list(query, original)
        f = find_in_list(query, final)
        if not o or not f:
            return 0.5
        o_qty = original[original["product_name"] == o[0].product_name]["quantity"].sum()
        f_qty = final[final["product_name"]   == f[0].product_name]["quantity"].sum()
        return 1.0 if f_qty < o_qty else 0.3

    if t == "replace":
        from_q = action.target_ar or action.target
        to_q   = action.replacement_ar or action.replacement
        orig_from = find_in_list(from_q, original)
        fin_from  = find_in_list(from_q, final)
        fin_to    = find_in_list(to_q,   final)
        # Success: from gone from final, to present in final
        from_gone = not fin_from or (orig_from and fin_from[0].score < 50)
        to_present = bool(fin_to and fin_to[0].score >= 50)
        if from_gone and to_present: return 1.0
        if to_present: return 0.7
        if from_gone:  return 0.4
        return 0.1

    if t == "keep_only":
        cat    = action.category
        if not cat:
            return 0.5
        final_cat  = final[final["category"] == cat]
        orig_cat   = original[original["category"] == cat]
        allowed_ar = action.allowed_ar or action.allowed
        # Count how many non-allowed items remain
        non_allowed = 0
        for _, row in final_cat.iterrows():
            hits = [find_in_list(q, final_cat) for q in allowed_ar]
            matched = any(h and h[0].product_name == row["product_name"] for h in hits if h)
            if not matched:
                non_allowed += 1
        return 1.0 if non_allowed == 0 else max(0.0, 1.0 - non_allowed * 0.2)

    if t == "preference":
        pref = action.preference.lower()
        if pref == "cheap":
            o_avg = original["unit_price"].mean() if not original.empty else 0
            f_avg = final["unit_price"].mean()   if not final.empty    else 0
            return 1.0 if f_avg <= o_avg else 0.4
        if pref == "quality":
            o_avg = original["unit_price"].mean() if not original.empty else 0
            f_avg = final["unit_price"].mean()   if not final.empty    else 0
            return 1.0 if f_avg >= o_avg else 0.4
        if pref in ("healthy", "diet", "gym"):
            # Check that vegetable/protein qty went up
            o_veg = original[original["category"]=="Vegetables"]["quantity"].sum()
            f_veg = final[final["category"]=="Vegetables"]["quantity"].sum()
            o_prot = original[original["category"]=="Proteins"]["quantity"].sum()
            f_prot = final[final["category"]=="Proteins"]["quantity"].sum()
            if pref == "gym":
                return 1.0 if f_prot >= o_prot else 0.4
            return 1.0 if f_veg >= o_veg else 0.5

    if t in ("add", "conditional"):
        # Just check list grew
        return 1.0 if len(final) > len(original) else 0.5

    return 0.7  # unknown type — neutral
