"""
apply_engine.py
════════════════════════════════════════════════════════════════════════
Safe, rule-enforced engine that applies NLP actions to the shopping list.

Hard rules:
  1. NEVER delete full category unless scope=category AND confidence ≥ 0.85
  2. NEVER remove essential items (chicken/eggs/milk/bread) unless explicit
  3. NEVER create duplicates
  4. ALWAYS keep diversity (min 1 item per category after modification)
  5. ALWAYS respect budget
  6. confidence < 0.7 → apply best guess but log warning

Public API:
  apply_actions(shopping_list, actions, dataset, budget) → ApplyResult
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from fuzzy_matcher import (
    find_in_list, find_in_dataset, resolve_category,
    norm, RowMatch, CONFIDENT, LOW_CONF,
)
from nlp_engine import Action, KNOWN_CATEGORIES

# ── Essential items that require explicit confirmation to delete ───────────────
ESSENTIAL_SLOTS = {
    "chicken", "eggs", "milk", "rice", "bread", "oil",
    "fish", "meat", "tomato", "onion", "potato",
}

BABY_KEYWORDS = [
    "حفاض", "pampers", "بامبرز", "بيبي فلاي", "سيريلاك",
    "هيرو جوديز", "بيبي بافس",
]


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class ActionResult:
    action_type: str
    target:      str
    message_ar:  str
    success:     bool
    match_logs:  list[dict] = field(default_factory=list)
    warning:     str        = ""


@dataclass
class ApplyResult:
    shopping_list: pd.DataFrame
    results:       list[ActionResult] = field(default_factory=list)
    baby_mode:     bool = False
    diet_mode:     bool = False
    gym_mode:      bool = False

    @property
    def messages(self) -> list[str]:
        return [r.message_ar for r in self.results]

    @property
    def match_logs(self) -> list[dict]:
        return [log for r in self.results for log in r.match_logs]

    @property
    def summary(self) -> str:
        return "\n".join(self.messages)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ensure_norm(df: pd.DataFrame) -> pd.DataFrame:
    if "product_name_norm" not in df.columns:
        df = df.copy()
        df["product_name_norm"] = df["product_name"].apply(norm)
    return df


def _used_norms(df: pd.DataFrame) -> set:
    return set(_ensure_norm(df)["product_name_norm"].tolist())


def _total_cost(df: pd.DataFrame) -> float:
    return float(df["total_price"].sum())


def _is_essential(product_name: str) -> bool:
    """True if this product matches an essential slot."""
    n = norm(product_name)
    for slot in ESSENTIAL_SLOTS:
        from fuzzy_matcher import _resolve_concept
        for synonym in _resolve_concept(slot):
            if synonym and synonym in n:
                return True
    return False


def _build_new_row(prod: pd.Series, quantity: int, slot: str = "added") -> dict:
    """Builds a shopping list row dict from a dataset product."""
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


# ── Individual handlers ───────────────────────────────────────────────────────

def _apply_add(
    df: pd.DataFrame,
    action: Action,
    dataset: pd.DataFrame,
    budget_left: float,
) -> ActionResult:
    """Adds a new product or category to the list."""
    query    = action.target_ar or action.target
    cat      = action.category or resolve_category(query)
    used     = _used_norms(df)
    logs: list[dict] = []

    match = find_in_dataset(
        query, dataset,
        category=cat, exclude_norms=used,
        max_price=budget_left,
    )

    if match is None:
        # Relax category
        match = find_in_dataset(query, dataset, exclude_norms=used, max_price=budget_left)

    if match is None or match.rejected:
        return ActionResult("add", query,
                            f"⚠️ لم أجد '{query}' في المنتجات المتاحة.", False)

    logs.append(match.to_log())
    prod    = match.row
    qty     = 1
    new_row = _build_new_row(prod, qty, slot="added")

    if new_row["total_price"] > budget_left:
        return ActionResult("add", query,
                            f"⚠️ '{prod['product_name'][:40]}' يتجاوز الميزانية المتبقية.",
                            False, logs)

    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    conf_note = f" (ثقة: {match.score:.0f}%)" if not match.confident else ""
    return ActionResult(
        "add", prod["product_name"],
        f"➕ تمت إضافة '{prod['product_name'][:45]}'{conf_note}",
        True, logs,
    )


def _apply_remove(
    df: pd.DataFrame,
    action: Action,
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, ActionResult]:
    query    = action.target_ar or action.target
    scope    = action.scope
    conf     = action.confidence
    logs: list[dict] = []

    # ── Category remove — only if explicit ────────────────────────────────────
    if scope == "category" and conf >= 0.85:
        cat = action.category or resolve_category(query)
        if not cat:
            hits = find_in_list(query, df, scope="category")
            cat = hits[0].row["category"] if hits else None

        if cat:
            n_before = len(df)
            new_df   = df[df["category"] != cat].reset_index(drop=True)
            removed  = n_before - len(new_df)
            return new_df, ActionResult(
                "remove", cat,
                f"🗑️ حُذف {removed} منتج من فئة '{cat}'",
                True, logs,
            )
        return df, ActionResult("remove", query, f"⚠️ لم أتعرف على الفئة '{query}'.", False)

    # ── Product remove — fuzzy match ──────────────────────────────────────────
    hits = find_in_list(query, df, scope="product")
    if not hits or hits[0].rejected:
        # Try Arabic then English
        hits2 = find_in_list(action.target, df, scope="product")
        hits  = hits2 if hits2 and (not hits or hits2[0].score > hits[0].score) else hits

    if not hits or hits[0].rejected:
        return df, ActionResult("remove", query,
                                f"⚠️ لم أجد '{query}' في القائمة.", False, logs)

    best = hits[0]
    logs.append(best.to_log())

    # Safety: warn if essential and confidence not super high
    if _is_essential(best.product_name) and conf < 0.90:
        warning = f"⚠️ '{best.product_name[:35]}' عنصر أساسي — تم الحذف بناءً على طلبك."
    else:
        warning = ""

    # Confidence gate
    if best.score < LOW_CONF:
        return df, ActionResult("remove", query,
                                f"⚠️ تطابق ضعيف ({best.score:.0f}%) لـ '{query}' — لم يُحذف.",
                                False, logs, warning=f"أعلى تطابق: {best.product_name[:40]}")

    mask   = df["product_name"] == best.product_name
    new_df = df[~mask].reset_index(drop=True)
    removed = int(mask.sum())

    conf_note = f" (ثقة: {best.score:.0f}%)" if not best.confident else ""
    return new_df, ActionResult(
        "remove", best.product_name,
        f"🗑️ حُذف {removed} × '{best.product_name[:45]}'{conf_note}",
        True, logs, warning=warning,
    )


def _apply_replace(
    df: pd.DataFrame,
    action: Action,
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, ActionResult]:
    from_q   = action.target_ar or action.target
    to_q     = action.replacement_ar or action.replacement
    logs: list[dict] = []

    # Find the item to replace in list
    from_hits = find_in_list(from_q, df)
    if not from_hits or from_hits[0].rejected:
        from_hits2 = find_in_list(action.target, df)
        if from_hits2 and (not from_hits or from_hits2[0].score > from_hits[0].score):
            from_hits = from_hits2

    if not from_hits or from_hits[0].score < LOW_CONF:
        return df, ActionResult("replace", from_q,
                                f"⚠️ لم أجد '{from_q}' في القائمة.", False, logs)

    best_from = from_hits[0]
    logs.append({**best_from.to_log(), "role": "from"})

    avg_qty  = int(best_from.row["quantity"])
    cat      = best_from.row["category"]
    used     = _used_norms(df) - {norm(best_from.product_name)}

    # Remove old item
    new_df = df[df["product_name"] != best_from.product_name].copy().reset_index(drop=True)

    # Find replacement in dataset (try Arabic first — more accurate for Arabic db)
    replacement: Optional[RowMatch] = None
    for q in [to_q, action.replacement]:
        r = find_in_dataset(q, dataset, category=cat, exclude_norms=used)
        if r and r.score >= LOW_CONF:
            replacement = r
            break

    # Relax category
    if replacement is None:
        for q in [to_q, action.replacement]:
            r = find_in_dataset(q, dataset, exclude_norms=used)
            if r and r.score >= LOW_CONF:
                replacement = r
                break

    if replacement is None:
        msg = (f"🗑️ حُذف '{best_from.product_name[:35]}' "
               f"⚠️ لم أجد بديلاً مناسباً لـ '{to_q}'.")
        return new_df, ActionResult("replace", from_q, msg, False, logs)

    logs.append({**replacement.to_log(), "role": "to"})
    new_row = _build_new_row(replacement.row, avg_qty, slot="replaced")
    new_df  = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    conf_note = f" (ثقة: {replacement.score:.0f}%)" if not replacement.confident else ""
    return new_df, ActionResult(
        "replace", best_from.product_name,
        f"🔄 '{best_from.product_name[:35]}' → '{replacement.row['product_name'][:35]}'{conf_note}",
        True, logs,
    )


def _apply_increase(
    df: pd.DataFrame,
    action: Action,
    budget: float,
    dataset: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, ActionResult]:
    """
    Delegates to smart_modify.smart_increase for intelligent mode-based logic.
    Falls back to pure quantity if dataset is unavailable.
    """
    if dataset is not None:
        from smart_modify import smart_increase
        new_df, result = smart_increase(df, action, dataset, budget)
        return new_df, ActionResult(
            action_type = "increase",
            target      = action.target_ar or action.target,
            message_ar  = result.message_ar,
            success     = result.success,
            match_logs  = result.match_logs,
            warning     = f"[{result.strategy_used}] {result.reason}" if result.success else "",
        )

    # Fallback: pure quantity (no dataset available)
    query  = action.target_ar or action.target
    scope  = action.scope
    factor = action.amount
    logs: list[dict] = []

    if scope == "category":
        cat = action.category or resolve_category(query)
        if not cat:
            return df, ActionResult("increase", query, f"⚠️ لم أتعرف على الفئة '{query}'.", False)
        mask = df["category"] == cat
    else:
        hits = find_in_list(query, df)
        if not hits or hits[0].score < LOW_CONF:
            return df, ActionResult("increase", query, f"⚠️ لم أجد '{query}'.", False, logs)
        logs.append(hits[0].to_log())
        mask = df["product_name"] == hits[0].product_name

    if not mask.any():
        return df, ActionResult("increase", query, f"⚠️ لم أجد '{query}' في القائمة.", False, logs)

    new_df = df.copy()
    new_df.loc[mask, "quantity"] = (
        new_df.loc[mask, "quantity"] * factor
    ).apply(lambda x: max(1, round(x)))
    new_df.loc[mask, "total_price"] = (
        new_df.loc[mask, "quantity"] * new_df.loc[mask, "unit_price"]
    )
    if _total_cost(new_df) > budget:
        scale = budget / _total_cost(new_df)
        new_df.loc[mask, "quantity"] = (
            new_df.loc[mask, "quantity"] * scale
        ).apply(lambda x: max(1, round(x)))
        new_df.loc[mask, "total_price"] = new_df.loc[mask, "quantity"] * new_df.loc[mask, "unit_price"]
    pct = round((factor - 1) * 100)
    return new_df, ActionResult("increase", query, f"⬆️ '{query}' +{pct}% ({int(mask.sum())} منتج)", True, logs)


def _apply_decrease(
    df: pd.DataFrame,
    action: Action,
    dataset: pd.DataFrame | None = None,
    budget: float = float("inf"),
) -> tuple[pd.DataFrame, ActionResult]:
    """
    Delegates to smart_modify.smart_decrease for intelligent mode-based logic.
    """
    if dataset is not None:
        from smart_modify import smart_decrease
        new_df, result = smart_decrease(df, action, dataset, budget)
        return new_df, ActionResult(
            action_type = "decrease",
            target      = action.target_ar or action.target,
            message_ar  = result.message_ar,
            success     = result.success,
            match_logs  = result.match_logs,
            warning     = f"[{result.strategy_used}] {result.reason}" if result.success else "",
        )

    # Fallback
    query  = action.target_ar or action.target
    scope  = action.scope
    factor = action.amount
    logs: list[dict] = []

    if scope == "category":
        cat = action.category or resolve_category(query)
        if not cat:
            return df, ActionResult("decrease", query, f"⚠️ لم أتعرف على الفئة '{query}'.", False)
        mask = df["category"] == cat
    else:
        hits = find_in_list(query, df)
        if not hits or hits[0].score < LOW_CONF:
            return df, ActionResult("decrease", query, f"⚠️ لم أجد '{query}'.", False, logs)
        logs.append(hits[0].to_log())
        mask = df["product_name"] == hits[0].product_name

    if not mask.any():
        return df, ActionResult("decrease", query, f"⚠️ لم أجد '{query}' في القائمة.", False, logs)

    new_df = df.copy()
    new_df.loc[mask, "quantity"] = (
        new_df.loc[mask, "quantity"] * factor
    ).apply(lambda x: max(1, round(x)))
    new_df.loc[mask, "total_price"] = new_df.loc[mask, "quantity"] * new_df.loc[mask, "unit_price"]
    pct = round((1 - factor) * 100)
    return new_df, ActionResult("decrease", query, f"⬇️ '{query}' -{pct}% ({int(mask.sum())} منتج)", True, logs)


def _apply_keep_only(
    df: pd.DataFrame,
    action: Action,
) -> tuple[pd.DataFrame, ActionResult]:
    """
    Removes everything in a category EXCEPT the allowed items.
    Uses fuzzy matching to identify allowed items (never keyword-only).
    """
    cat = action.category or resolve_category(action.target or action.target_ar)
    if not cat:
        return df, ActionResult("keep_only", action.target,
                                f"⚠️ لم أتعرف على الفئة للاحتفاظ.", False)

    cat_rows  = df[df["category"] == cat]
    allowed_q = list(action.allowed_ar) + list(action.allowed)
    logs: list[dict] = []
    keep_names: set[str] = set()

    for q in allowed_q:
        if not q.strip():
            continue
        hits = find_in_list(q, cat_rows)
        if hits and hits[0].score >= LOW_CONF:
            keep_names.add(hits[0].product_name)
            logs.append({**hits[0].to_log(), "keep_query": q})

    if not keep_names:
        return df, ActionResult(
            "keep_only", cat,
            f"⚠️ لم أجد المنتجات المطلوب الاحتفاظ بها في '{cat}'.",
            False, logs,
        )

    not_cat  = df["category"] != cat
    allowed  = df["product_name"].isin(keep_names)
    new_df   = df[not_cat | allowed].reset_index(drop=True)
    removed  = len(df) - len(new_df)
    kept_str = "، ".join(list(keep_names)[:4])

    return new_df, ActionResult(
        "keep_only", cat,
        f"🔒 '{cat}': احتُفظ بـ [{kept_str}]، حُذف {removed} منتج",
        True, logs,
    )


def _apply_preference(
    df: pd.DataFrame,
    action: Action,
    dataset: pd.DataFrame,
    budget: float,
) -> tuple[pd.DataFrame, ActionResult]:
    pref = action.preference.lower()
    logs: list[dict] = []

    # ── Healthy / Diet ────────────────────────────────────────────────────────
    if pref in ("healthy", "diet"):
        rules = [
            Action(type="increase", scope="category", category="Vegetables",
                   target="Vegetables", target_ar="الخضار", amount=1.4),
            Action(type="increase", scope="category", category="Fruits",
                   target="Fruits", target_ar="الفاكهة", amount=1.4),
            Action(type="decrease", scope="category", category="Snacks",
                   target="Snacks", target_ar="السناكس", amount=0.5),
            Action(type="decrease", scope="category", category="Oils & Fats",
                   target="Oils & Fats", target_ar="الزيوت", amount=0.7),
        ]
        msgs = []
        for r in rules:
            if r.type == "increase":
                df, res = _apply_increase(df, r, budget, dataset=dataset)
            else:
                df, res = _apply_decrease(df, r, dataset=dataset, budget=budget)
            logs.extend(res.match_logs)
            msgs.append(res.message_ar)
        icon = "🥗" if pref == "diet" else "🥦"
        return df, ActionResult("preference", pref,
                                f"{icon} {pref.capitalize()}:\n" + "\n".join(msgs), True, logs)

    # ── Gym ───────────────────────────────────────────────────────────────────
    if pref == "gym":
        rules = [
            Action(type="increase", scope="category", category="Proteins",
                   target="Proteins", target_ar="البروتينات", amount=1.8),
            Action(type="decrease", scope="category", category="Snacks",
                   target="Snacks", target_ar="السناكس", amount=0.4),
            Action(type="decrease", scope="category", category="Grains",
                   target="Grains", target_ar="النشويات", amount=0.7),
        ]
        msgs = []
        for r in rules:
            if r.type == "increase":
                df, res = _apply_increase(df, r, budget, dataset=dataset)
            else:
                df, res = _apply_decrease(df, r, dataset=dataset, budget=budget)
            logs.extend(res.match_logs)
            msgs.append(res.message_ar)
        return df, ActionResult("preference", pref,
                                "💪 جيم:\n" + "\n".join(msgs), True, logs)

    # ── Cheap / Quality ───────────────────────────────────────────────────────
    if pref in ("cheap", "quality"):
        from utils import PRICE_THRESHOLDS
        prefer_cheap   = pref == "cheap"
        prefer_quality = pref == "quality"
        replaced = 0
        df = _ensure_norm(df.copy())
        used = _used_norms(df)

        for idx, row in df.iterrows():
            cat    = row["category"]
            thresh = PRICE_THRESHOLDS.get(cat, {"cheap": 50, "quality": 150})
            price  = row["unit_price"]

            if prefer_cheap   and price <= thresh["cheap"]:   continue
            if prefer_quality and price >= thresh["quality"]: continue

            current_norm = row.get("product_name_norm", norm(row["product_name"]))
            excl = used - {current_norm}

            match = find_in_dataset(
                row["product_name"], dataset, category=cat,
                prefer_cheap=prefer_cheap, prefer_quality=prefer_quality,
                exclude_norms=excl,
            )
            if match is None:
                continue
            if prefer_cheap   and match.row["effective_price"] >= price: continue
            if prefer_quality and match.row["effective_price"] <= price: continue

            prod = match.row
            df.at[idx, "product_name"]      = prod["product_name"]
            df.at[idx, "product_name_norm"] = prod.get("product_name_norm", norm(prod["product_name"]))
            df.at[idx, "unit_price"]        = prod["effective_price"]
            df.at[idx, "total_price"]       = round(prod["effective_price"] * row["quantity"], 2)
            df.at[idx, "slot"]              = "preference_swap"
            df.at[idx, "discount_pct"]      = prod.get("discount_pct", 0)
            used.add(prod.get("product_name_norm", norm(prod["product_name"])))
            logs.append(match.to_log())
            replaced += 1

        label = "💰 الأرخص" if prefer_cheap else "⭐ الأجود"
        return df, ActionResult("preference", pref,
                                f"{label}: {replaced} منتج تم تبديله", True, logs)

    return df, ActionResult("preference", pref, f"⚠️ تفضيل غير معروف: {pref}", False)


def _apply_baby_mode(
    df: pd.DataFrame,
    dataset: pd.DataFrame,
    budget_left: float,
) -> tuple[pd.DataFrame, ActionResult]:
    """Adds diapers + baby food + baby drinks."""
    slots = [
        {"kws": ["حفاض","pampers","بامبرز"],   "cat": "Cleaning & Personal Care", "n": 1},
        {"kws": ["سيريلاك","هيرو","بيبي"],     "cat": None,                       "n": 2},
        {"kws": ["ويف","مشروب اطفال","عصير"],  "cat": "Beverages",                "n": 1},
    ]
    used = _used_norms(df)
    added_rows = []
    msgs: list[str] = []

    for slot in slots:
        pool = dataset.copy()
        if slot["cat"]:
            pool = pool[pool["category"] == slot["cat"]]
        pool = _ensure_norm(pool)
        pool = pool[~pool["product_name_norm"].isin(used)]

        mask = pd.Series([False] * len(pool), index=pool.index)
        for kw in slot["kws"]:
            mask |= pool["product_name"].str.contains(kw, case=False, na=False, regex=False)

        matched = pool[mask].nsmallest(slot["n"], "effective_price")
        for _, prod in matched.iterrows():
            row = _build_new_row(prod, 1, slot="baby")
            if row["total_price"] <= budget_left:
                added_rows.append(row)
                used.add(prod.get("product_name_norm", norm(prod["product_name"])))
                budget_left -= row["total_price"]
                msgs.append(f"  👶 {prod['product_name'][:45]} — {prod['effective_price']:.1f}ج.م")

    if not added_rows:
        return df, ActionResult("baby_mode", "baby_products",
                                "⚠️ لم أجد منتجات أطفال مناسبة.", False)

    new_df = pd.concat([df, pd.DataFrame(added_rows)], ignore_index=True)
    return new_df, ActionResult(
        "baby_mode", "baby_products",
        f"👶 {len(added_rows)} منتج أطفال:\n" + "\n".join(msgs),
        True,
    )


def _apply_conditional(
    df: pd.DataFrame,
    action: Action,
    dataset: pd.DataFrame,
    budget: float,
) -> tuple[pd.DataFrame, ActionResult]:
    """Handles lifestyle-context actions (baby, gym already handled, etc.)."""
    pref = action.preference.lower()
    if "baby" in pref:
        return _apply_baby_mode(df, dataset, budget - _total_cost(df))
    # Fallback to preference engine
    action.preference = pref
    return _apply_preference(df, action, dataset, budget)


# ── Main entry ────────────────────────────────────────────────────────────────
def apply_actions(
    shopping_list: pd.DataFrame,
    actions: list[Action],
    dataset: pd.DataFrame,
    budget: float = float("inf"),
    baby_mode: bool = False,
    diet_mode: bool = False,
    gym_mode: bool = False,
) -> ApplyResult:
    """
    Applies all NLP actions to shopping_list sequentially.
    Never lets the list become invalid.

    Returns ApplyResult with the updated list and all messages/logs.
    """
    df      = _ensure_norm(shopping_list.copy())
    results: list[ActionResult] = []

    # ── Mode triggers ─────────────────────────────────────────────────────────
    if baby_mode:
        df, res = _apply_baby_mode(df, dataset, budget - _total_cost(df))
        results.append(res)

    if diet_mode and not any(a.preference == "diet" for a in actions):
        a = Action(type="preference", scope="global", preference="diet")
        df, res = _apply_preference(df, a, dataset, budget)
        results.append(res)

    if gym_mode and not any(a.preference == "gym" for a in actions):
        a = Action(type="preference", scope="global", preference="gym")
        df, res = _apply_preference(df, a, dataset, budget)
        results.append(res)

    # ── Action loop ───────────────────────────────────────────────────────────
    for action in actions:
        t = action.type

        if t == "add":
            res = _apply_add(df, action, dataset, budget - _total_cost(df))
            if res.success:
                # Fetch the actual new_df (add returns only result, not df)
                # Rerun to get df update
                used  = _used_norms(df)
                q     = action.target_ar or action.target
                cat   = action.category or None
                match = find_in_dataset(q, dataset, category=cat,
                                        exclude_norms=used,
                                        max_price=budget - _total_cost(df))
                if match and not match.rejected:
                    new_row = _build_new_row(match.row, 1, slot="added")
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        elif t == "remove":
            df, res = _apply_remove(df, action, dataset)

        elif t == "replace":
            df, res = _apply_replace(df, action, dataset)

        elif t == "increase":
            df, res = _apply_increase(df, action, budget, dataset=dataset)

        elif t == "decrease":
            df, res = _apply_decrease(df, action, dataset=dataset, budget=budget)

        elif t == "keep_only":
            df, res = _apply_keep_only(df, action)

        elif t == "preference":
            df, res = _apply_preference(df, action, dataset, budget)

        elif t == "conditional":
            df, res = _apply_conditional(df, action, dataset, budget)

        else:
            res = ActionResult(t, "", f"⚠️ نوع أمر غير معروف: {t}", False)

        results.append(res)

    # ── Safety: never return empty list ───────────────────────────────────────
    if df.empty or len(df) < 3:
        # Revert to original
        df = _ensure_norm(shopping_list.copy())
        results.append(ActionResult(
            "safety", "",
            "⚠️ التعديل كان سيُفرغ القائمة — تمت استعادة النسخة الأصلية.",
            False,
        ))

    return ApplyResult(
        shopping_list=df.reset_index(drop=True),
        results=results,
        baby_mode=baby_mode,
        diet_mode=diet_mode,
        gym_mode=gym_mode,
    )
