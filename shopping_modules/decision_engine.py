"""
decision_engine.py  — Safe Execution & Validation Layer
══════════════════════════════════════════════════════════════════════════════
Receives a DecisionPlan from nlp_engine and executes it safely.

Three validation passes before any modification:
  1. Product existence check  — does the named product exist in list/dataset?
  2. Duplicate prevention     — don't add something already present
  3. Budget enforcement       — don't exceed total budget

Then executes the actions:
  add               → find exact or fuzzy match in dataset, insert row
  increase_quantity → raise qty of matched list row
  decrease_quantity → lower qty (min 1)
  remove            → delete matched rows
  replace           → remove old, add new

Returns ExecutionResult with: updated list, messages, validation log, debug.

PUBLIC API:
  execute_plan(plan, shopping_list, dataset, budget) → ExecutionResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from fuzzy_matcher import (
    find_in_list, find_in_dataset, norm,
    RowMatch, CONFIDENT, LOW_CONF,
)
from nlp_engine import DecisionPlan, PlannedAction, KNOWN_CATEGORIES

# ── Thresholds ─────────────────────────────────────────────────────────────────
EXECUTION_MIN_CONFIDENCE = 0.55    # skip action if confidence < this
FUZZY_MATCH_MIN_SCORE    = 50      # minimum fuzzy score to accept a match
ESSENTIAL_PRODUCTS = {             # never remove without very high confidence
    "chicken","eggs","milk","rice","bread","oil","fish","meat","tomato","onion","potato"
}


# ── Result containers ─────────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    passed:     bool
    reason:     str
    matched_name: str = ""
    score:        float = 0.0


@dataclass
class ActionLog:
    action_type: str
    target:      str
    outcome:     str          # "applied" | "skipped" | "failed"
    message_ar:  str
    validation:  ValidationResult
    debug:       dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    shopping_list: pd.DataFrame
    action_logs:   list[ActionLog] = field(default_factory=list)
    baby_mode:     bool = False
    diet_mode:     bool = False
    gym_mode:      bool = False
    intent:        str  = ""
    strategy:      str  = ""
    diversity_score: float = 0.5
    analysis_reason: str  = ""
    strategy_reason: str  = ""

    @property
    def messages(self) -> list[str]:
        return [log.message_ar for log in self.action_logs if log.message_ar]

    @property
    def applied_count(self) -> int:
        return sum(1 for l in self.action_logs if l.outcome == "applied")

    @property
    def skipped_count(self) -> int:
        return sum(1 for l in self.action_logs if l.outcome == "skipped")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ensure_norm(df: pd.DataFrame) -> pd.DataFrame:
    if "product_name_norm" not in df.columns:
        df = df.copy()
        df["product_name_norm"] = df["product_name"].apply(norm)
    return df


def _total_cost(df: pd.DataFrame) -> float:
    return float(df["total_price"].sum()) if not df.empty else 0.0


def _used_norms(df: pd.DataFrame) -> set:
    return set(_ensure_norm(df)["product_name_norm"].tolist())


def _is_essential(product_name: str) -> bool:
    n = norm(product_name)
    from fuzzy_matcher import CONCEPTS
    for slot in ESSENTIAL_PRODUCTS:
        if slot in CONCEPTS:
            if any(norm(s) in n for s in CONCEPTS[slot]):
                return True
    return False


def _build_row(prod: pd.Series, quantity: int, slot: str = "ai_added") -> dict:
    return {
        "category":          prod["category"],
        "product_name":      prod["product_name"],
        "product_name_norm": prod.get("product_name_norm", norm(prod["product_name"])),
        "unit_price":        prod["effective_price"],
        "quantity":          max(1, quantity),
        "total_price":       round(prod["effective_price"] * max(1, quantity), 2),
        "slot":              slot,
        "source":            prod.get("source", "Carrefour Egypt"),
        "discount_pct":      prod.get("discount_pct", 0),
    }


# ── Per-action validators ─────────────────────────────────────────────────────
def _validate_add(
    action: PlannedAction,
    shopping_list: pd.DataFrame,
    dataset: pd.DataFrame,
    budget_left: float,
) -> tuple[ValidationResult, Optional[pd.Series]]:
    """
    Validates an 'add' action:
    1. Finds the product in dataset (exact then fuzzy)
    2. Checks it's not already in the list
    3. Checks budget allows it
    Returns (ValidationResult, matched dataset row or None)
    """
    if action.confidence < EXECUTION_MIN_CONFIDENCE:
        return ValidationResult(False, f"ثقة منخفضة جداً ({action.confidence:.0%})"), None

    # Try exact match first
    exact = dataset[dataset["product_name"] == action.product]
    if not exact.empty:
        prod = exact.iloc[0]
        score = 100.0
    else:
        # Fuzzy fallback — use strict threshold to prevent hallucination acceptance
        match = find_in_dataset(action.product, dataset,
                                category=action.category or None)
        if match is None or match.score < max(FUZZY_MATCH_MIN_SCORE, 60):
            return ValidationResult(
                False,
                f"لم أجد '{action.product}' في قاعدة البيانات",
            ), None
        prod  = match.row
        score = match.score

    # Duplicate check
    existing = find_in_list(prod["product_name"], shopping_list)
    if existing and existing[0].score >= 80:
        return ValidationResult(
            False,
            f"'{prod['product_name'][:40]}' موجود بالفعل في القائمة",
            matched_name=prod["product_name"], score=score,
        ), None

    # Budget check
    cost = prod["effective_price"] * action.quantity
    if cost > budget_left + 5:   # 5 EGP tolerance
        return ValidationResult(
            False,
            f"'{prod['product_name'][:35]}' ({cost:.0f} ج.م) يتجاوز الميزانية المتبقية ({budget_left:.0f} ج.م)",
            matched_name=prod["product_name"], score=score,
        ), None

    return ValidationResult(
        True, "OK", matched_name=prod["product_name"], score=score
    ), prod


def _validate_modify(
    action: PlannedAction,
    shopping_list: pd.DataFrame,
    action_type: str,
) -> tuple[ValidationResult, Optional[RowMatch]]:
    """Validates increase/decrease/remove — product must exist in list."""
    if action.confidence < EXECUTION_MIN_CONFIDENCE:
        return ValidationResult(False, f"ثقة منخفضة ({action.confidence:.0%})"), None

    # Exact match
    exact = shopping_list[shopping_list["product_name"] == action.product]
    if not exact.empty:
        match = RowMatch(row=exact.iloc[0], score=100.0,
                         method="exact", matched_term=action.product)
    else:
        hits = find_in_list(action.product, shopping_list)
        if not hits or hits[0].score < FUZZY_MATCH_MIN_SCORE:
            return ValidationResult(
                False,
                f"لم أجد '{action.product}' في القائمة الحالية",
            ), None
        match = hits[0]

    # Extra guard: confirm before removing essential items
    if action_type == "remove" and _is_essential(match.product_name) and action.confidence < 0.88:
        return ValidationResult(
            False,
            f"'{match.product_name[:40]}' عنصر أساسي — يحتاج ثقة أعلى للحذف (حالياً {action.confidence:.0%})",
            matched_name=match.product_name, score=match.score,
        ), None

    return ValidationResult(
        True, "OK", matched_name=match.product_name, score=match.score
    ), match


# ── Action executors ──────────────────────────────────────────────────────────
def _exec_add(
    df: pd.DataFrame,
    action: PlannedAction,
    dataset: pd.DataFrame,
    budget_left: float,
) -> tuple[pd.DataFrame, ActionLog]:
    v, prod = _validate_add(action, df, dataset, budget_left)

    if not v.passed:
        return df, ActionLog(
            "add", action.product, "skipped",
            f"⚠️ تجاوزت الإضافة: {v.reason}",
            v, {"confidence": action.confidence},
        )

    new_row = _build_row(prod, action.quantity, slot="ai_added")
    df      = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    conf_n  = f" (ثقة: {v.score:.0f}%)" if v.score < 100 else ""
    return df, ActionLog(
        "add", v.matched_name, "applied",
        f"➕ أُضيف: '{v.matched_name[:50]}' × {action.quantity}{conf_n}\n   💡 {action.reason}",
        v, {"confidence": action.confidence, "qty": action.quantity},
    )


def _exec_increase(
    df: pd.DataFrame,
    action: PlannedAction,
) -> tuple[pd.DataFrame, ActionLog]:
    v, match = _validate_modify(action, df, "increase")

    if not v.passed or match is None:
        return df, ActionLog(
            "increase", action.product, "skipped",
            f"⚠️ تجاوزت الزيادة: {v.reason}", v,
        )

    df = df.copy()
    mask = df["product_name"] == match.product_name
    df.loc[mask, "quantity"]    = df.loc[mask, "quantity"] + action.amount
    df.loc[mask, "total_price"] = df.loc[mask, "quantity"] * df.loc[mask, "unit_price"]
    conf_n = f" (ثقة: {v.score:.0f}%)" if v.score < 100 else ""

    return df, ActionLog(
        "increase", match.product_name, "applied",
        f"⬆️ زيادة '{match.product_name[:45]}' +{action.amount}{conf_n}\n   💡 {action.reason}",
        v, {"amount": action.amount},
    )


def _exec_decrease(
    df: pd.DataFrame,
    action: PlannedAction,
) -> tuple[pd.DataFrame, ActionLog]:
    v, match = _validate_modify(action, df, "decrease")

    if not v.passed or match is None:
        return df, ActionLog(
            "decrease", action.product, "skipped",
            f"⚠️ تجاوزت التقليل: {v.reason}", v,
        )

    df   = df.copy()
    mask = df["product_name"] == match.product_name
    new_qty = (df.loc[mask, "quantity"] - action.amount).apply(lambda x: max(1, x))
    df.loc[mask, "quantity"]    = new_qty
    df.loc[mask, "total_price"] = df.loc[mask, "quantity"] * df.loc[mask, "unit_price"]
    conf_n = f" (ثقة: {v.score:.0f}%)" if v.score < 100 else ""

    return df, ActionLog(
        "decrease", match.product_name, "applied",
        f"⬇️ تقليل '{match.product_name[:45]}' -{action.amount}{conf_n}\n   💡 {action.reason}",
        v, {"amount": action.amount},
    )


def _exec_remove(
    df: pd.DataFrame,
    action: PlannedAction,
) -> tuple[pd.DataFrame, ActionLog]:
    v, match = _validate_modify(action, df, "remove")

    if not v.passed or match is None:
        return df, ActionLog(
            "remove", action.product, "skipped",
            f"⚠️ تجاوزت الحذف: {v.reason}", v,
        )

    df      = df[df["product_name"] != match.product_name].reset_index(drop=True)
    conf_n  = f" (ثقة: {v.score:.0f}%)" if v.score < 100 else ""

    return df, ActionLog(
        "remove", match.product_name, "applied",
        f"🗑️ حُذف: '{match.product_name[:50]}'{conf_n}\n   💡 {action.reason}",
        v,
    )


def _exec_replace(
    df: pd.DataFrame,
    action: PlannedAction,
    dataset: pd.DataFrame,
    budget_left: float,
) -> tuple[pd.DataFrame, ActionLog]:
    v_from, match = _validate_modify(action, df, "remove")
    if not v_from.passed or match is None:
        return df, ActionLog(
            "replace", action.product, "skipped",
            f"⚠️ لم أجد '{action.product}': {v_from.reason}", v_from,
        )

    # Save qty before removing
    qty = int(df[df["product_name"] == match.product_name]["quantity"].sum())

    # Remove old
    df = df[df["product_name"] != match.product_name].copy().reset_index(drop=True)

    # Find replacement
    # Try exact match in dataset first
    repl_exact = dataset[dataset["product_name"] == action.replacement]
    if not repl_exact.empty:
        repl_prod = repl_exact.iloc[0]
        repl_score = 100.0
    else:
        excl  = _used_norms(df)
        repl_match = find_in_dataset(
            action.replacement, dataset,
            category=action.category or match.row["category"],
            exclude_norms=excl,
        )
        if repl_match is None or repl_match.score < FUZZY_MATCH_MIN_SCORE:
            # Can't find replacement — restore original
            df = pd.concat([df, pd.DataFrame([{
                "category": match.row["category"],
                "product_name": match.product_name,
                "product_name_norm": norm(match.product_name),
                "unit_price": match.row["unit_price"],
                "quantity": qty,
                "total_price": match.row["unit_price"] * qty,
                "slot": match.row.get("slot",""),
                "source": match.row.get("source",""),
                "discount_pct": match.row.get("discount_pct",0),
            }])], ignore_index=True)
            return df, ActionLog(
                "replace", action.product, "skipped",
                f"⚠️ لم أجد '{action.replacement}' كبديل", v_from,
            )
        repl_prod  = repl_match.row
        repl_score = repl_match.score

    # Budget check
    add_cost = repl_prod["effective_price"] * qty
    if add_cost > budget_left + 20:
        qty = max(1, int((budget_left + 20) // repl_prod["effective_price"]))

    new_row = _build_row(repl_prod, qty, slot="replaced")
    df      = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    conf_n  = f" (ثقة: {repl_score:.0f}%)" if repl_score < 100 else ""
    return df, ActionLog(
        "replace", match.product_name, "applied",
        f"🔄 '{match.product_name[:35]}' → '{repl_prod['product_name'][:35]}'{conf_n}\n   💡 {action.reason}",
        ValidationResult(True, "OK", repl_prod["product_name"], repl_score),
        {"old_qty": qty},
    )


# ── Mode-triggered actions (baby/diet/gym) ────────────────────────────────────
def _exec_baby_mode(
    df: pd.DataFrame,
    dataset: pd.DataFrame,
    budget_left: float,
) -> tuple[pd.DataFrame, ActionLog]:
    slots = [
        {"kws": ["حفاض","pampers","بامبرز","بيبي فلاي"],
         "cat": "Cleaning & Personal Care", "n": 1},
        {"kws": ["سيريلاك","هيرو","بيبي"],
         "cat": None, "n": 2},
        {"kws": ["ويف","مشروب اطفال"],
         "cat": "Beverages", "n": 1},
    ]
    used     = _used_norms(df)
    added    = []
    msgs     = []
    df       = df.copy()

    for slot in slots:
        pool = dataset.copy()
        if slot["cat"]:
            pool = pool[pool["category"] == slot["cat"]]
        pool = _ensure_norm(pool)
        pool = pool[~pool["product_name_norm"].isin(used)]

        mask = pd.Series([False]*len(pool), index=pool.index)
        for kw in slot["kws"]:
            mask |= pool["product_name"].str.contains(kw, case=False, na=False, regex=False)

        matched = pool[mask].nsmallest(slot["n"], "effective_price")
        for _, prod in matched.iterrows():
            cost = prod["effective_price"]
            if cost > budget_left:
                continue
            row  = _build_row(prod, 1, slot="baby")
            df   = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            used.add(prod.get("product_name_norm", norm(prod["product_name"])))
            budget_left -= cost
            added.append(prod["product_name"])
            msgs.append(f"  👶 {prod['product_name'][:45]} — {cost:.0f}ج.م")

    if not added:
        return df, ActionLog("baby_mode","baby","skipped","⚠️ لم أجد منتجات أطفال",
                             ValidationResult(False,"no baby products"))
    return df, ActionLog("baby_mode","baby","applied",
                         f"👶 {len(added)} منتج أطفال:\n" + "\n".join(msgs),
                         ValidationResult(True,"OK"))


def _exec_preference(
    df: pd.DataFrame,
    dataset: pd.DataFrame,
    budget: float,
    pref: str,
) -> tuple[pd.DataFrame, ActionLog]:
    """Applies global preference (healthy/diet/gym/cheap/quality) via smart_modify."""
    from nlp_engine import Action as LegacyAction
    from apply_engine import _apply_preference

    legacy_action = LegacyAction(
        type="preference", scope="global", preference=pref,
        raw={"preference": pref},
    )
    new_df, result = _apply_preference(df, legacy_action, dataset, budget)
    return new_df, ActionLog(
        "preference", pref, "applied",
        result.message_ar, ValidationResult(True, "OK"),
    )


# ── New V9 handlers ───────────────────────────────────────────────────────────

def _exec_update_size(
    df: pd.DataFrame,
    action: PlannedAction,
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, ActionLog]:
    """
    Swaps a product for a larger/more efficient package size.
    E.g.: 10 × 320g rice → 2 × 1kg rice
    """
    v, match = _validate_modify(action, df, "replace")
    if not v.passed or match is None:
        return df, ActionLog("update_size", action.product, "skipped",
                             f"⚠️ لم أجد '{action.product}': {v.reason}", v)

    old_qty   = int(df[df["product_name"] == match.product_name]["quantity"].sum())
    old_price = float(df[df["product_name"] == match.product_name]["unit_price"].iloc[0])

    # Find better-sized version in dataset
    from consumption_planner import parse_package_size
    target_name = action.new_size or action.product

    pool = dataset[dataset["category"] == match.row["category"]].copy()
    pool = pool[~pool["product_name"].isin(df["product_name"].tolist())]

    # Parse sizes and find ones with larger package
    current_size = parse_package_size(match.product_name)
    current_val  = current_size[0] if current_size else 0

    best_upgrade = None
    best_ratio   = 0.0

    for _, row in pool.iterrows():
        s = parse_package_size(row["product_name"])
        if s and s[0] > current_val:
            # Calculate price efficiency
            ratio = s[0] / row["effective_price"]   # g per j.m
            if ratio > best_ratio:
                best_ratio   = ratio
                best_upgrade = row

    if best_upgrade is None:
        return df, ActionLog("update_size", action.product, "skipped",
                             f"⚠️ لم أجد حزمة أكبر لـ '{action.product}'",
                             ValidationResult(False, "no larger size"))

    from consumption_planner import parse_package_size as pps
    new_s = pps(best_upgrade["product_name"])
    total_needed  = current_val * old_qty
    new_qty       = max(1, -(-int(total_needed) // max(1, int(new_s[0] if new_s else 1))))

    # Remove old, add new
    df      = df[df["product_name"] != match.product_name].copy().reset_index(drop=True)
    new_row = _build_row(best_upgrade, new_qty, slot="size_optimized")
    df      = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    old_total = old_price * old_qty
    new_total = best_upgrade["effective_price"] * new_qty
    saved     = old_total - new_total

    return df, ActionLog(
        "update_size", match.product_name, "applied",
        f"📦 تحسين الحجم: {old_qty}×'{match.product_name[:35]}' → "
        f"{new_qty}×'{best_upgrade['product_name'][:35]}'\n"
        f"   💰 وفرت: {saved:.0f}ج.م | {action.reason}",
        ValidationResult(True, "OK", best_upgrade["product_name"], 90.0),
        {"old_qty": old_qty, "new_qty": new_qty, "saved": round(saved, 1)},
    )


def _exec_rebalance(
    df: pd.DataFrame,
    dataset: pd.DataFrame,
    budget: float,
    family_size: int = 4,
    days: int = 30,
) -> tuple[pd.DataFrame, list[ActionLog]]:
    """
    Rebalances the entire list against consumption targets.
    Uses optimize_existing_sizes + adds missing essentials.
    """
    from consumption_planner import optimize_existing_sizes, compute_targets, find_optimal_packages
    from fuzzy_matcher import norm as fnorm

    logs: list[ActionLog] = []

    # Step 1: Size optimization
    df, changes = optimize_existing_sizes(df, dataset, family_size, days)
    for c in changes:
        logs.append(ActionLog("update_size", c["item"], "applied",
                              f"📦 {c['old']} → {c['new']} (وفرت {c['saved']:.0f}ج.م)",
                              ValidationResult(True,"OK")))

    # Step 2: Fill consumption gaps
    targets   = compute_targets(family_size, days)
    used      = set(df["product_name_norm"].tolist() if "product_name_norm" in df.columns
                    else df["product_name"].apply(fnorm).tolist())
    budget_left = max(0, budget - float(df["total_price"].sum()))

    for item, target in targets.items():
        # Check if category is already covered
        cat_rows = df[df["category"] == target.category]
        kw_mask  = pd.Series([False]*len(cat_rows), index=cat_rows.index)
        for kw in target.keywords:
            kw_mask |= cat_rows["product_name"].str.contains(fnorm(kw), case=False, na=False, regex=False)
        if kw_mask.any():
            continue  # already have this item

        # Find optimal package
        choices = find_optimal_packages(target, dataset, budget_left * 0.1, used)
        if not choices:
            continue

        best = choices[0]
        if best.total_price > budget_left:
            continue

        new_row = _build_row(best.dataset_row, best.quantity, slot=f"rebalance_{item}")
        df      = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        used.add(fnorm(best.product_name))
        budget_left -= best.total_price

        logs.append(ActionLog("add", item, "applied",
                              f"➕ أُضيف (توازن): '{best.product_name[:45]}' ×{best.quantity}",
                              ValidationResult(True,"OK")))

    if not logs:
        logs.append(ActionLog("rebalance","list","applied",
                              "✅ القائمة متوازنة — لا تغييرات جوهرية مطلوبة",
                              ValidationResult(True,"OK")))

    return df.reset_index(drop=True), logs


# ── Main execution entry ──────────────────────────────────────────────────────
def execute_plan(
    plan:          DecisionPlan,
    shopping_list: pd.DataFrame,
    dataset:       pd.DataFrame,
    budget:        float = float("inf"),
    family_size:   int   = 4,
    days:          int   = 30,
) -> ExecutionResult:
    """
    Executes a DecisionPlan safely against the shopping list.

    Flow per action:
      1. Validate (confidence, existence, budget, duplicates)
      2. Execute if valid
      3. Skip with log if invalid

    Returns ExecutionResult with updated list + full audit trail.
    """
    df   = _ensure_norm(shopping_list.copy())
    logs: list[ActionLog] = []

    # ── Mode triggers (baby / diet / gym) ─────────────────────────────────────
    if plan.baby_mode:
        budget_left = max(0, budget - _total_cost(df))
        df, log = _exec_baby_mode(df, dataset, budget_left)
        logs.append(log)

    if plan.diet_mode and plan.strategy != "preference":
        df, log = _exec_preference(df, dataset, budget, "diet")
        logs.append(log)

    if plan.gym_mode and plan.strategy != "preference":
        df, log = _exec_preference(df, dataset, budget, "gym")
        logs.append(log)

    # ── Action execution loop ─────────────────────────────────────────────────
    for action in plan.actions:
        budget_left = max(0, budget - _total_cost(df))

        if action.type == "add":
            df, log = _exec_add(df, action, dataset, budget_left)

        elif action.type == "increase_quantity":
            df, log = _exec_increase(df, action)

        elif action.type == "decrease_quantity":
            df, log = _exec_decrease(df, action)

        elif action.type == "remove":
            df, log = _exec_remove(df, action)

        elif action.type == "replace":
            budget_left = max(0, budget - _total_cost(df))
            df, log = _exec_replace(df, action, dataset, budget_left)

        elif action.type == "update_size":
            df, log = _exec_update_size(df, action, dataset)

        elif action.type == "rebalance":
            df, extra_logs = _exec_rebalance(df, dataset, budget, family_size, days)
            logs.extend(extra_logs)
            continue   # logs already appended

        else:
            log = ActionLog(action.type, action.product, "skipped",
                            f"⚠️ نوع غير معروف: {action.type}",
                            ValidationResult(False, "unknown type"))

        logs.append(log)

    # ── Handle preference or rebalance strategy ───────────────────────────────
    if plan.strategy == "rebalance" and not plan.actions:
        df, extra_logs = _exec_rebalance(df, dataset, budget, family_size, days)
        logs.extend(extra_logs)
    elif plan.strategy == "preference" and not plan.actions:
        # Determine which preference from intent
        intent_lower = plan.intent.lower()
        pref = ("healthy" if "healthy" in intent_lower or "diet" in intent_lower else
                "cheap"   if "cheap"   in intent_lower else
                "quality" if "quality" in intent_lower else
                "gym"     if "gym"     in intent_lower else "healthy")
        df, log = _exec_preference(df, dataset, budget, pref)
        logs.append(log)

    # ── Safety: never return empty list ───────────────────────────────────────
    if df.empty or len(df) < 3:
        df = _ensure_norm(shopping_list.copy())
        logs.append(ActionLog("safety", "", "skipped",
                              "⚠️ التعديل كان سيُفرغ القائمة — تمت الاستعادة",
                              ValidationResult(False, "list would be empty")))

    return ExecutionResult(
        shopping_list  = df.reset_index(drop=True),
        action_logs    = logs,
        baby_mode      = plan.baby_mode,
        diet_mode      = plan.diet_mode,
        gym_mode       = plan.gym_mode,
        intent         = plan.intent,
        strategy       = plan.strategy,
        diversity_score= plan.diversity_score,
        analysis_reason= plan.analysis_reason,
        strategy_reason= plan.strategy_reason,
    )
