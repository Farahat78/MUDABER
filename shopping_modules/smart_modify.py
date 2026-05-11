"""
smart_modify.py
══════════════════════════════════════════════════════════════════════
Smart Increase / Decrease Engine

The old system blindly multiplied quantities. This module thinks like
a human: deciding WHAT to change, not just HOW MUCH.

MODES
─────
quantity  → only raise/lower quantities of existing items
variety   → only add/remove product variety (new items, same category)
hybrid    → do both: add new items AND increase quantities
auto      → system decides intelligently based on:
              1. current category diversity (items count)
              2. available budget headroom
              3. average quantity already in the category

PUBLIC API
──────────
smart_increase(shopping_list, action, dataset, budget) → (df, result)
smart_decrease(shopping_list, action, dataset, budget) → (df, result)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from fuzzy_matcher import (
    find_in_list, find_in_dataset, resolve_category,
    norm, RowMatch, CONFIDENT, LOW_CONF,
)
from nlp_engine import Action, KNOWN_CATEGORIES

# ── Thresholds ─────────────────────────────────────────────────────────────────
LOW_DIVERSITY_THRESHOLD   = 3   # ≤ 3 items in category → needs more variety
IDEAL_DIVERSITY_THRESHOLD = 6   # target items per category
MIN_BUDGET_TO_ADD         = 20  # EGP — minimum headroom needed to add a product
MAX_QTY_BEFORE_VARIETY    = 5   # if avg qty > 5, prefer variety over more qty


# ── Decision result ────────────────────────────────────────────────────────────
@dataclass
class ModifyDecision:
    mode:         str          # quantity | variety | hybrid
    reason:       str          # Arabic explanation for UI
    qty_factor:   float        # multiplier for existing quantities (1.0 = no change)
    items_to_add: int          # how many new items to add
    items_to_drop:int          # for decrease: how many low-priority items to remove
    debug:        dict         = field(default_factory=dict)


@dataclass
class ModifyResult:
    shopping_list:  pd.DataFrame
    message_ar:     str
    strategy_used:  str
    reason:         str
    items_added:    list[str]   = field(default_factory=list)
    items_removed:  list[str]   = field(default_factory=list)
    qty_changed:    int         = 0
    match_logs:     list[dict]  = field(default_factory=list)
    debug:          dict        = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.items_added or self.items_removed or self.qty_changed)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _ensure_norm(df: pd.DataFrame) -> pd.DataFrame:
    if "product_name_norm" not in df.columns:
        df = df.copy()
        df["product_name_norm"] = df["product_name"].apply(norm)
    return df


def _category_stats(df: pd.DataFrame, category: str) -> dict:
    """Returns diversity/qty stats for a category in the shopping list."""
    cat_df = df[df["category"] == category]
    if cat_df.empty:
        return {"count": 0, "total_qty": 0, "avg_qty": 0.0,
                "total_spend": 0.0, "avg_price": 0.0}
    return {
        "count":       len(cat_df),
        "total_qty":   int(cat_df["quantity"].sum()),
        "avg_qty":     float(cat_df["quantity"].mean()),
        "total_spend": float(cat_df["total_price"].sum()),
        "avg_price":   float(cat_df["unit_price"].mean()),
    }


def _budget_headroom(df: pd.DataFrame, total_budget: float) -> float:
    """Returns how much EGP is still available."""
    return max(0.0, total_budget - float(df["total_price"].sum()))


def _used_norms(df: pd.DataFrame) -> set:
    return set(_ensure_norm(df)["product_name_norm"].tolist())


def _pick_new_items(
    category:      str,
    dataset:       pd.DataFrame,
    exclude_norms: set,
    budget_per_item: float,
    n:             int,
    diverse:       bool = True,
) -> list[pd.Series]:
    """
    Picks n NEW products from dataset for a category.
    - Excludes items already in the list
    - Respects per-item budget cap
    - Picks diverse items (not all from same sub-category)
    """
    pool = dataset[
        (dataset["category"] == category) &
        (dataset["effective_price"] <= budget_per_item)
    ].copy()
    pool = _ensure_norm(pool)
    pool = pool[~pool["product_name_norm"].isin(exclude_norms)]

    if pool.empty:
        return []

    if diverse:
        # Sort by sub-category then pick from different sub-cats
        pool["_rank"] = pool.groupby("sub_category")["effective_price"].rank()
        pool = pool.sort_values(["sub_category", "effective_price"])
        # Round-robin across sub-categories
        selected: list[pd.Series] = []
        sub_cats = list(pool["sub_category"].unique())
        random.shuffle(sub_cats)
        for sc in sub_cats:
            if len(selected) >= n:
                break
            sc_pool = pool[pool["sub_category"] == sc]
            if not sc_pool.empty:
                selected.append(sc_pool.iloc[0])
        # If not enough, pad with cheapest remaining
        if len(selected) < n:
            already = {s["product_name"] for s in selected}
            remaining = pool[~pool["product_name"].isin(already)].nsmallest(n, "effective_price")
            selected += [remaining.iloc[i] for i in range(min(n - len(selected), len(remaining)))]
        return selected[:n]
    else:
        # Just pick n cheapest
        top = pool.nsmallest(min(n * 3, len(pool)), "effective_price")
        return [top.iloc[i] for i in range(min(n, len(top)))]


# ── Auto decision engine ───────────────────────────────────────────────────────
def decide_increase_mode(
    shopping_list: pd.DataFrame,
    category:      str,
    budget:        float,
    hint_mode:     str = "auto",
) -> ModifyDecision:
    """
    Decides how to handle an increase request.
    Returns a ModifyDecision with mode, qty_factor, items_to_add, and reason.
    """
    if hint_mode != "auto":
        # User (or Gemini) specified a mode explicitly
        if hint_mode == "quantity":
            return ModifyDecision(
                mode="quantity", reason="المستخدم طلب زيادة الكمية فقط",
                qty_factor=1.5, items_to_add=0, items_to_drop=0,
            )
        if hint_mode == "variety":
            return ModifyDecision(
                mode="variety", reason="المستخدم طلب تنويع المنتجات",
                qty_factor=1.0, items_to_add=2, items_to_drop=0,
            )
        if hint_mode == "hybrid":
            return ModifyDecision(
                mode="hybrid", reason="المستخدم طلب تنويع وزيادة",
                qty_factor=1.3, items_to_add=1, items_to_drop=0,
            )

    # ── Auto mode: intelligent decision ──────────────────────────────────────
    stats      = _category_stats(shopping_list, category)
    headroom   = _budget_headroom(shopping_list, budget)
    item_count = stats["count"]
    avg_qty    = stats["avg_qty"]

    debug = {
        "category_items":   item_count,
        "avg_qty":          round(avg_qty, 1),
        "budget_headroom":  round(headroom, 0),
        "low_diversity":    item_count <= LOW_DIVERSITY_THRESHOLD,
        "high_qty":         avg_qty >= MAX_QTY_BEFORE_VARIETY,
        "can_add":          headroom >= MIN_BUDGET_TO_ADD,
    }

    # Case 1: Very few items + enough budget → variety (add new products)
    if item_count <= LOW_DIVERSITY_THRESHOLD and headroom >= MIN_BUDGET_TO_ADD:
        return ModifyDecision(
            mode="variety",
            reason=f"التنوع منخفض ({item_count} منتجات) والميزانية كافية — إضافة منتجات جديدة",
            qty_factor=1.0,
            items_to_add=min(3, max(1, IDEAL_DIVERSITY_THRESHOLD - item_count)),
            items_to_drop=0,
            debug=debug,
        )

    # Case 2: Moderate diversity + enough budget + qty already high → hybrid
    if headroom >= MIN_BUDGET_TO_ADD * 2 and avg_qty >= MAX_QTY_BEFORE_VARIETY:
        return ModifyDecision(
            mode="hybrid",
            reason=f"الكميات كافية (متوسط {avg_qty:.0f}) — إضافة منتج جديد مع زيادة طفيفة",
            qty_factor=1.2,
            items_to_add=1,
            items_to_drop=0,
            debug=debug,
        )

    # Case 3: Good diversity but low qty → pure quantity increase
    if item_count > LOW_DIVERSITY_THRESHOLD and avg_qty < MAX_QTY_BEFORE_VARIETY:
        return ModifyDecision(
            mode="quantity",
            reason=f"التنوع جيد ({item_count} منتجات) — زيادة الكميات",
            qty_factor=1.5,
            items_to_add=0,
            items_to_drop=0,
            debug=debug,
        )

    # Case 4: Both low diversity AND low qty + budget → hybrid
    if headroom >= MIN_BUDGET_TO_ADD:
        return ModifyDecision(
            mode="hybrid",
            reason="تنوع ضعيف وكميات منخفضة — إضافة وزيادة",
            qty_factor=1.3,
            items_to_add=min(2, max(1, IDEAL_DIVERSITY_THRESHOLD - item_count)),
            items_to_drop=0,
            debug=debug,
        )

    # Default: just increase quantities (no budget for new items)
    return ModifyDecision(
        mode="quantity",
        reason="الميزانية محدودة — زيادة الكميات فقط",
        qty_factor=1.4,
        items_to_add=0,
        items_to_drop=0,
        debug=debug,
    )


def decide_decrease_mode(
    shopping_list: pd.DataFrame,
    category:      str,
    budget:        float,
    hint_mode:     str = "auto",
) -> ModifyDecision:
    """Decides how to handle a decrease request."""
    if hint_mode == "quantity":
        return ModifyDecision(
            mode="quantity", reason="تقليل الكميات",
            qty_factor=0.6, items_to_add=0, items_to_drop=0,
        )
    if hint_mode == "variety":
        return ModifyDecision(
            mode="variety", reason="حذف بعض المنتجات ذات الأولوية المنخفضة",
            qty_factor=1.0, items_to_add=0, items_to_drop=2,
        )
    if hint_mode == "hybrid":
        return ModifyDecision(
            mode="hybrid", reason="تقليل الكميات وحذف بعض المنتجات",
            qty_factor=0.7, items_to_add=0, items_to_drop=1,
        )

    # Auto
    stats      = _category_stats(shopping_list, category)
    item_count = stats["count"]
    avg_qty    = stats["avg_qty"]

    debug = {
        "category_items": item_count,
        "avg_qty":        round(avg_qty, 1),
    }

    # Many items + high qty → hybrid (remove some + reduce qty of rest)
    if item_count > IDEAL_DIVERSITY_THRESHOLD and avg_qty > MAX_QTY_BEFORE_VARIETY:
        return ModifyDecision(
            mode="hybrid",
            reason=f"كثير من المنتجات والكميات — حذف بعضها وتقليل الباقي",
            qty_factor=0.7, items_to_add=0,
            items_to_drop=max(1, item_count - IDEAL_DIVERSITY_THRESHOLD),
            debug=debug,
        )

    # Only a few items → just reduce qty (don't remove items, might break essentials)
    if item_count <= LOW_DIVERSITY_THRESHOLD:
        return ModifyDecision(
            mode="quantity",
            reason=f"عدد المنتجات قليل ({item_count}) — تقليل الكميات فقط",
            qty_factor=0.6, items_to_add=0, items_to_drop=0,
            debug=debug,
        )

    # Normal case
    return ModifyDecision(
        mode="quantity",
        reason="تقليل الكميات",
        qty_factor=0.6, items_to_add=0, items_to_drop=0,
        debug=debug,
    )


# ── Low-priority item selector (for decrease variety/hybrid) ──────────────────
def _lowest_priority_items(
    cat_df: pd.DataFrame,
    n:      int,
) -> list[str]:
    """
    Returns n product names with lowest priority to remove.
    Priority score = unit_price (cheap & single-qty items removed first).
    Essential items (chicken, eggs, milk, etc.) are protected.
    """
    from apply_engine import _is_essential   # late import to avoid circular

    scored = cat_df.copy()
    scored["_priority"] = (
        scored["unit_price"] * 0.4 +
        scored["quantity"]   * 0.6
    )

    # Sort ascending (lowest priority first)
    scored = scored.sort_values("_priority")
    candidates = []
    for _, row in scored.iterrows():
        if not _is_essential(row["product_name"]):
            candidates.append(row["product_name"])
        if len(candidates) >= n:
            break
    return candidates


# ── Main smart_increase ───────────────────────────────────────────────────────
def smart_increase(
    shopping_list: pd.DataFrame,
    action:        Action,
    dataset:       pd.DataFrame,
    budget:        float = float("inf"),
) -> tuple[pd.DataFrame, ModifyResult]:
    """
    Intelligently increases a category or product in the shopping list.

    Flow:
    1. Resolve target → category
    2. Decide mode (auto / quantity / variety / hybrid)
    3. Execute: add new items and/or increase quantities
    4. Return updated list + rich ModifyResult
    """
    df    = _ensure_norm(shopping_list.copy())
    query = action.target_ar or action.target
    scope = action.scope
    mode  = getattr(action, "mode", "auto") if hasattr(action, "mode") else \
            action.raw.get("mode", "auto")
    logs: list[dict] = []
    items_added:   list[str] = []
    qty_changed:   int       = 0

    # ── Resolve target ────────────────────────────────────────────────────────
    if scope == "category":
        category = action.category or resolve_category(query)
        if not category:
            return df, ModifyResult(df, f"⚠️ لم أتعرف على الفئة '{query}'.", "none", "", [])
        mask = df["category"] == category
        target_name = category
    else:
        hits = find_in_list(query, df)
        if not hits or hits[0].score < LOW_CONF:
            hits2 = find_in_list(action.target, df)
            hits  = hits2 if (hits2 and (not hits or hits2[0].score > hits[0].score)) else hits
        if not hits or hits[0].score < LOW_CONF:
            return df, ModifyResult(df, f"⚠️ لم أجد '{query}' في القائمة.", "none", "", [])

        logs.append(hits[0].to_log())
        target_name = hits[0].product_name
        category    = hits[0].row["category"]
        mask        = df["product_name"] == target_name

    # ── Decide mode ───────────────────────────────────────────────────────────
    decision = decide_increase_mode(df, category, budget, hint_mode=mode)
    headroom = _budget_headroom(df, budget)

    # ── Execute: quantity part ────────────────────────────────────────────────
    if decision.qty_factor > 1.0 and mask.any():
        df.loc[mask, "quantity"] = (
            df.loc[mask, "quantity"] * decision.qty_factor
        ).apply(lambda x: max(1, round(x)))
        df.loc[mask, "total_price"] = (
            df.loc[mask, "quantity"] * df.loc[mask, "unit_price"]
        )

        # Budget guard after qty change
        if df["total_price"].sum() > budget:
            scale = budget / df["total_price"].sum()
            df.loc[mask, "quantity"] = (
                df.loc[mask, "quantity"] * scale
            ).apply(lambda x: max(1, round(x)))
            df.loc[mask, "total_price"] = (
                df.loc[mask, "quantity"] * df.loc[mask, "unit_price"]
            )

        qty_changed = int(mask.sum())

    # ── Execute: variety part (add new items) ─────────────────────────────────
    if decision.items_to_add > 0 and headroom >= MIN_BUDGET_TO_ADD:
        used       = _used_norms(df)
        headroom   = _budget_headroom(df, budget)
        # Estimate affordable price per new item
        price_cap  = min(headroom / max(1, decision.items_to_add), headroom * 0.8)

        new_prods = _pick_new_items(
            category      = category,
            dataset       = dataset,
            exclude_norms = used,
            budget_per_item = price_cap,
            n             = decision.items_to_add,
            diverse       = True,
        )

        for prod in new_prods:
            qty = max(1, round(
                # Scale qty to roughly match existing category avg
                _category_stats(df, category)["avg_qty"]
            ))
            new_row = {
                "category":          prod["category"],
                "product_name":      prod["product_name"],
                "product_name_norm": prod.get("product_name_norm", norm(prod["product_name"])),
                "unit_price":        prod["effective_price"],
                "quantity":          qty,
                "total_price":       round(prod["effective_price"] * qty, 2),
                "slot":              "added_variety",
                "source":            prod.get("source", "Carrefour"),
                "discount_pct":      prod.get("discount_pct", 0),
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            items_added.append(prod["product_name"])
            logs.append({"added": prod["product_name"], "price": prod["effective_price"],
                         "category": category})

    # ── Build message ─────────────────────────────────────────────────────────
    parts: list[str] = []
    mode_icons = {"quantity": "⬆️", "variety": "🆕", "hybrid": "⬆️🆕", "none": "⚠️"}
    icon = mode_icons.get(decision.mode, "⬆️")

    if qty_changed:
        pct = round((decision.qty_factor - 1) * 100)
        parts.append(f"⬆️ رُفعت كمية {qty_changed} منتج في '{target_name}' بنسبة {pct}%")

    if items_added:
        parts.append(
            f"🆕 أُضيف {len(items_added)} منتج جديد في '{category}':\n"
            + "\n".join(f"  • {p[:50]}" for p in items_added)
        )

    if not parts:
        parts.append(f"⚠️ لم يتغير شيء في '{target_name}'")

    message = "\n".join(parts)

    return df.reset_index(drop=True), ModifyResult(
        shopping_list = df.reset_index(drop=True),
        message_ar    = message,
        strategy_used = decision.mode,
        reason        = decision.reason,
        items_added   = items_added,
        items_removed = [],
        qty_changed   = qty_changed,
        match_logs    = logs,
        debug         = {
            **decision.debug,
            "strategy": decision.mode,
            "reason":   decision.reason,
            "qty_factor": decision.qty_factor,
            "items_added": len(items_added),
        },
    )


# ── Main smart_decrease ───────────────────────────────────────────────────────
def smart_decrease(
    shopping_list: pd.DataFrame,
    action:        Action,
    dataset:       pd.DataFrame,
    budget:        float = float("inf"),
) -> tuple[pd.DataFrame, ModifyResult]:
    """
    Intelligently decreases a category or product.
    May reduce quantities, remove low-priority items, or both.
    """
    df    = _ensure_norm(shopping_list.copy())
    query = action.target_ar or action.target
    scope = action.scope
    mode  = action.raw.get("mode", "auto")
    logs: list[dict] = []
    items_removed: list[str] = []
    qty_changed:   int       = 0

    # ── Resolve target ────────────────────────────────────────────────────────
    if scope == "category":
        category = action.category or resolve_category(query)
        if not category:
            return df, ModifyResult(df, f"⚠️ لم أتعرف على الفئة '{query}'.", "none", "", [])
        mask = df["category"] == category
        target_name = category
    else:
        hits = find_in_list(query, df)
        if not hits or hits[0].score < LOW_CONF:
            hits2 = find_in_list(action.target, df)
            hits = hits2 if (hits2 and (not hits or hits2[0].score > hits[0].score)) else hits
        if not hits or hits[0].score < LOW_CONF:
            return df, ModifyResult(df, f"⚠️ لم أجد '{query}' في القائمة.", "none", "", [])

        logs.append(hits[0].to_log())
        target_name = hits[0].product_name
        category    = hits[0].row["category"]
        mask        = df["product_name"] == target_name

    # ── Decide mode ───────────────────────────────────────────────────────────
    decision = decide_decrease_mode(df, category, budget, hint_mode=mode)

    # ── Execute: remove low-priority items ────────────────────────────────────
    if decision.items_to_drop > 0:
        cat_df    = df[mask] if scope != "category" else df[df["category"] == category]
        to_remove = _lowest_priority_items(cat_df, decision.items_to_drop)

        for pname in to_remove:
            remove_mask = df["product_name"] == pname
            if remove_mask.any():
                df = df[~remove_mask].reset_index(drop=True)
                items_removed.append(pname)
                logs.append({"removed": pname})

        # Rebuild mask after removals
        if scope == "category":
            mask = df["category"] == category
        else:
            mask = df["product_name"] == target_name

    # ── Execute: quantity part ────────────────────────────────────────────────
    if decision.qty_factor < 1.0 and mask.any():
        df.loc[mask, "quantity"] = (
            df.loc[mask, "quantity"] * decision.qty_factor
        ).apply(lambda x: max(1, round(x)))
        df.loc[mask, "total_price"] = (
            df.loc[mask, "quantity"] * df.loc[mask, "unit_price"]
        )
        qty_changed = int(mask.sum())

    # ── Build message ─────────────────────────────────────────────────────────
    parts: list[str] = []

    if items_removed:
        parts.append(
            f"🗑️ حُذف {len(items_removed)} منتج ذو أولوية منخفضة:\n"
            + "\n".join(f"  • {p[:50]}" for p in items_removed)
        )

    if qty_changed:
        pct = round((1 - decision.qty_factor) * 100)
        parts.append(f"⬇️ خُفضت كمية {qty_changed} منتج في '{target_name}' بنسبة {pct}%")

    if not parts:
        parts.append(f"⚠️ لم يتغير شيء في '{target_name}'")

    message = "\n".join(parts)

    return df.reset_index(drop=True), ModifyResult(
        shopping_list = df.reset_index(drop=True),
        message_ar    = message,
        strategy_used = decision.mode,
        reason        = decision.reason,
        items_added   = [],
        items_removed = items_removed,
        qty_changed   = qty_changed,
        match_logs    = logs,
        debug         = {
            **decision.debug,
            "strategy":      decision.mode,
            "reason":        decision.reason,
            "qty_factor":    decision.qty_factor,
            "items_dropped": len(items_removed),
        },
    )
