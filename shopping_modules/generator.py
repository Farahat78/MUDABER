"""
generator.py
─────────────────────────────────────────────────────────────
Core generation engine.

Architecture:
  1. Mandatory Slot System   — guarantees essential items are always present
  2. Budget-aware filler     — adds complementary products within remaining budget
  3. Diversity enforcer      — prevents duplicates and ensures variety
  4. Quantity estimator      — realistic household amounts per family size
  5. Controlled randomness   — results differ across runs via weighted sampling
"""

import random
import re
import pandas as pd
import numpy as np

from utils import (
    CATEGORIES,
    BUDGET_ALLOCATIONS,
    NECESSITY_SCORES,
    budget_for_category,
    compute_price_stats,
    estimate_quantity,
    score_product,
    family_scale,
)

# ─── Mandatory item slots ─────────────────────────────────────────────────────
# Each slot defines the category + one or more Arabic/English keyword sets.
# The engine tries every keyword set in order until a match is found.

MANDATORY_SLOTS = [
    # ── Proteins ─────────────────────────────────────────────────────────────
    {"category": "Proteins",  "slot": "chicken",
     "keywords": ["فراخ", "دجاج", "chicken", "أجنحة دجاج", "صدر دجاج", "ران دجاج"],
     "neg_kw": ["مدخن", "لانشون", "بسطرمة", "بانيه"]},
    {"category": "Proteins",  "slot": "red_meat",
     "keywords": ["لحم بقري", "لحمة بقري", "بيف", "لحم غنم", "كبدة", "beef", "مكعبات لحم"],
     "neg_kw": ["مدخن", "لانشون", "بسطرمة"]},
    {"category": "Proteins",  "slot": "fish",
     "keywords": ["سمك", "تونا", "fish", "salmon", "بوري", "بلطي", "فيليه سمك"],
     "neg_kw": []},
    {"category": "Proteins",  "slot": "eggs",
     "keywords": ["بيض", "بيص", "egg"],
     "neg_kw": ["شوكولاتة", "كيك", "بسكويت"]},
    # ── Dairy ────────────────────────────────────────────────────────────────
    {"category": "Dairy",     "slot": "milk",
     "keywords": ["حليب", "لبن طازج", "لبن كامل", "full milk", "milk"],
     "neg_kw": ["زبادي", "كريمة", "شوكولاتة", "بالفراولة", "فراولة"]},
    {"category": "Dairy",     "slot": "white_cheese",
     "keywords": ["جبنة بيضاء", "جبنه بيضاء", "فيتا", "فيتة", "جبنة طرية"],
     "neg_kw": ["مخلل", "رومي", "شيدر", "كريمة", "مطبوخة"]},
    {"category": "Dairy",     "slot": "yogurt",
     "keywords": ["زبادي", "yogurt"],
     "neg_kw": ["مشروب", "drink"]},
    # ── Vegetables ───────────────────────────────────────────────────────────
    {"category": "Vegetables", "slot": "tomato",
     "keywords": ["طماطم", "tomato"],
     "neg_kw": ["صلصة", "معلب", "مخلل"]},
    {"category": "Vegetables", "slot": "potato",
     "keywords": ["بطاطس", "بطاطا", "potato"],
     "neg_kw": ["مقلية", "شيبس", "مجمد"]},
    {"category": "Vegetables", "slot": "onion",
     "keywords": ["بصل", "onion"],
     "neg_kw": ["مخلل", "مجفف"]},
    {"category": "Vegetables", "slot": "cucumber",
     "keywords": ["خيار", "cucumber"],
     "neg_kw": ["مخلل"]},
    {"category": "Vegetables", "slot": "carrot",
     "keywords": ["جزر", "carrot"],
     "neg_kw": ["عصير"]},
    # ── Grains ───────────────────────────────────────────────────────────────
    {"category": "Grains",    "slot": "rice",
     "keywords": ["أرز", "ارز", "rice bag", "أرز مصري", "أرز بسمتي"],
     "neg_kw": ["كعك", "بسكويت", "حبوب", "دقيق", "سيريال"]},
    {"category": "Grains",    "slot": "pasta",
     "keywords": ["مكرونة", "باستا", "pasta", "spaghetti", "لسان عصفور", "بابيون"],
     "neg_kw": ["صوص", "sauce", "كيلو"]},
    {"category": "Grains",    "slot": "lentils",
     "keywords": ["عدس", "lentil", "فول", "حمص"],
     "neg_kw": []},
    # ── Beverages ────────────────────────────────────────────────────────────
    {"category": "Beverages", "slot": "tea",
     "keywords": ["شاي", "tea"],
     "neg_kw": ["آيس", "ice", "مثلج"]},
    {"category": "Beverages", "slot": "coffee",
     "keywords": ["قهوة", "coffee", "نسكافيه", "نسكافية"],
     "neg_kw": ["شوكولاتة كوكس"]},
    # ── Oils & Fats ──────────────────────────────────────────────────────────
    {"category": "Oils & Fats", "slot": "cooking_oil",
     "keywords": ["زيت نباتي", "زيت ذرة", "زيت زيتون", "زيت طبخ", "oil", "زيت دوار"],
     "neg_kw": ["سبراي", "spray"]},
    # ── Cleaning & Personal Care ─────────────────────────────────────────────
    {"category": "Cleaning & Personal Care", "slot": "detergent",
     "keywords": ["مسحوق غسيل", "منظف ملابس", "سائل غسيل", "detergent", "ابيض ناصع"],
     "neg_kw": []},
    {"category": "Cleaning & Personal Care", "slot": "shampoo",
     "keywords": ["شامبو", "shampoo"],
     "neg_kw": []},
    {"category": "Cleaning & Personal Care", "slot": "soap",
     "keywords": ["صابون استحمام", "صابون يدين", "soap", "جل استحمام"],
     "neg_kw": ["غسيل", "مسحوق"]},
    {"category": "Cleaning & Personal Care", "slot": "tissues",
     "keywords": ["مناديل", "tissue", "ورق تواليت", "مناديل حمام"],
     "neg_kw": ["مبللة", "wet"]},
]


def _keyword_match(name: str, keywords: list[str], neg_kw: list[str]) -> bool:
    """Returns True if name contains any keyword but none of the neg_kw."""
    name_l = name.lower()
    has_pos = any(kw.lower() in name_l for kw in keywords)
    has_neg = any(nk.lower() in name_l for nk in neg_kw) if neg_kw else False
    return has_pos and not has_neg


def _pick_mandatory(
    df: pd.DataFrame,
    slot: dict,
    used_names: set,
    lifestyle: str,
    budget_cap: float,
) -> pd.Series | None:
    """
    Picks the best affordable product for a mandatory slot.
    Prefers value-for-money; breaks ties with a small random offset.
    """
    cat_df = df[
        (df["category"] == slot["category"]) &
        (df["effective_price"] <= budget_cap) &
        (~df["product_name_norm"].isin(used_names))
    ].copy()

    # Filter by keywords
    mask = cat_df["product_name_norm"].apply(
        lambda n: _keyword_match(n, slot["keywords"], slot["neg_kw"])
    )
    matched = cat_df[mask]

    if matched.empty:
        # Relax: try without neg_kw filter
        mask2 = cat_df["product_name_norm"].apply(
            lambda n: any(kw.lower() in n.lower() for kw in slot["keywords"])
        )
        matched = cat_df[mask2]

    if matched.empty:
        return None

    # Sort by effective_price ascending then sample top-5 (controlled randomness)
    top = matched.nsmallest(min(5, len(matched)), "effective_price")
    weights = np.array([1 / (i + 1) for i in range(len(top))], dtype=float)
    weights /= weights.sum()
    chosen_idx = np.random.choice(top.index, p=weights)
    return df.loc[chosen_idx]


def _fill_category(
    df: pd.DataFrame,
    category: str,
    remaining_budget: float,
    used_names: set,
    lifestyle: str,
    max_items: int = 5,
    price_stats: dict | None = None,
) -> list[pd.Series]:
    """
    Fills a category up to remaining_budget using scored, diverse selection.
    """
    if remaining_budget <= 0 or max_items <= 0:
        return []

    cat_df = df[
        (df["category"] == category) &
        (df["effective_price"] <= remaining_budget) &
        (~df["product_name_norm"].isin(used_names))
    ].copy()

    if cat_df.empty:
        return []

    if price_stats is None:
        price_stats = compute_price_stats(df)

    # Score each candidate
    cat_df["_score"] = cat_df.apply(
        lambda r: score_product(r, NECESSITY_SCORES, price_stats), axis=1
    )

    # Weighted random sampling (top-20 candidates, weighted by score)
    pool = cat_df.nlargest(min(20, len(cat_df)), "_score")
    weights = np.array(pool["_score"].values, dtype=float)
    weights = np.clip(weights, 1e-9, None)
    weights /= weights.sum()

    selected = []
    spent = 0.0
    available_idx = list(pool.index)
    available_w = list(weights)

    attempts = 0
    while (
        len(selected) < max_items
        and available_idx
        and spent < remaining_budget
        and attempts < 50
    ):
        attempts += 1
        norm_w = np.array(available_w, dtype=float)
        norm_w /= norm_w.sum()

        idx = np.random.choice(available_idx, p=norm_w)
        row = df.loc[idx]

        if row["effective_price"] <= (remaining_budget - spent):
            selected.append(row)
            spent += row["effective_price"]

        # Remove from pool regardless (avoid re-trying expensive items)
        pos = available_idx.index(idx)
        available_idx.pop(pos)
        available_w.pop(pos)

    return selected


def generate_smart_shopping_list(
    df: pd.DataFrame,
    monthly_budget: float,
    family_size: int,
    lifestyle: str = "medium",
    random_seed: int | None = None,
    memory_skip_norms: set | None = None,
) -> pd.DataFrame:
    """
    Main generation function.

    Parameters
    ----------
    df            : cleaned DataFrame from preprocessing.load_and_clean()
    monthly_budget: total EGP budget for the month
    family_size   : number of people in the household
    lifestyle     : "low" | "medium" | "high"
    random_seed   : set for reproducible results (None = different each run)

    Returns
    -------
    DataFrame with columns:
        category, product_name, effective_price, quantity, total_price,
        unit, slot (mandatory or "filler"), source
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Apply memory: skip items user has consistently removed
    _memory_skip: set[str] = memory_skip_norms or set()

    alloc = BUDGET_ALLOCATIONS.get(lifestyle, BUDGET_ALLOCATIONS["medium"])
    price_stats = compute_price_stats(df)

    # Track per-category budget pools
    cat_budgets = {cat: monthly_budget * frac for cat, frac in alloc.items()}

    used_names: set[str] = set()
    list_rows: list[dict] = []

    # ── Phase 1: Fill mandatory slots ────────────────────────────────────────
    mandatory_by_cat: dict[str, float] = {}  # track spend per category

    for slot in MANDATORY_SLOTS:
        cat = slot["category"]
        cat_budget_cap = cat_budgets.get(cat, monthly_budget * 0.05)
        already_spent = mandatory_by_cat.get(cat, 0.0)
        available = cat_budget_cap - already_spent

        if available <= 2:
            continue

        product = _pick_mandatory(df, slot, used_names | _memory_skip, lifestyle, available)
        if product is None:
            continue

        qty = estimate_quantity(product["product_name_norm"], cat, family_size)
        total = round(product["effective_price"] * qty, 2)

        # If total exceeds available budget, reduce qty to 1
        if total > available:
            qty = 1
            total = round(product["effective_price"], 2)

        if total > available:
            continue  # truly can't afford — skip

        used_names.add(product["product_name_norm"])
        mandatory_by_cat[cat] = already_spent + total

        list_rows.append({
            "category":        cat,
            "product_name":    product["product_name"],
            "unit_price":      product["effective_price"],
            "quantity":        qty,
            "total_price":     total,
            "slot":            slot["slot"],
            "source":          product.get("source", "Carrefour Egypt"),
            "discount_pct":    product.get("discount_pct", 0),
        })

    # ── Phase 2: Fill remaining budget with diverse products ──────────────────
    for cat in CATEGORIES:
        spent_mandatory = mandatory_by_cat.get(cat, 0.0)
        remaining = cat_budgets.get(cat, 0.0) - spent_mandatory

        if remaining < 5:
            continue

        scale = family_scale(family_size)
        max_filler = max(2, int(5 * scale))  # more aggressive filling

        fillers = _fill_category(
            df, cat, remaining, used_names, lifestyle,
            max_items=max_filler, price_stats=price_stats
        )

        for product in fillers:
            qty = estimate_quantity(product["product_name_norm"], cat, family_size)
            total = round(product["effective_price"] * qty, 2)

            # Keep total within remaining budget
            if total > remaining:
                qty = max(1, int(remaining // product["effective_price"]))
                total = round(product["effective_price"] * qty, 2)

            if total > remaining or total <= 0:
                continue

            used_names.add(product["product_name_norm"])
            remaining -= total

            list_rows.append({
                "category":        cat,
                "product_name":    product["product_name"],
                "unit_price":      product["effective_price"],
                "quantity":        qty,
                "total_price":     total,
                "slot":            "filler",
                "source":          product.get("source", "Carrefour Egypt"),
                "discount_pct":    product.get("discount_pct", 0),
            })

    # ── Phase 3: Second pass — use any leftover budget across all categories ──
    total_spent = sum(r["total_price"] for r in list_rows)
    leftover = monthly_budget - total_spent

    if leftover > 20:
        # Redistribute leftover to top-priority categories
        priority_cats = ["Proteins", "Vegetables", "Dairy", "Grains"]
        extra_per_cat = leftover / len(priority_cats)

        for cat in priority_cats:
            if extra_per_cat < 5:
                break

            extra_fillers = _fill_category(
                df, cat, extra_per_cat, used_names, lifestyle,
                max_items=3, price_stats=price_stats
            )

            for product in extra_fillers:
                qty = estimate_quantity(product["product_name_norm"], cat, family_size)
                total = round(product["effective_price"] * qty, 2)
                
                if total > extra_per_cat:
                    qty = max(1, int(extra_per_cat // product["effective_price"]))
                    total = round(product["effective_price"] * qty, 2)
                
                if total > extra_per_cat or total <= 0:
                    continue

                used_names.add(product["product_name_norm"])
                extra_per_cat -= total

                list_rows.append({
                    "category":     cat,
                    "product_name": product["product_name"],
                    "unit_price":   product["effective_price"],
                    "quantity":     qty,
                    "total_price":  total,
                    "slot":         "filler",
                    "source":       product.get("source", "Carrefour Egypt"),
                    "discount_pct": product.get("discount_pct", 0),
                })

    result = pd.DataFrame(list_rows)
    if result.empty:
        return result

    result = result.sort_values(["category", "total_price"], ascending=[True, False])
    result = result.reset_index(drop=True)
    return result


def get_list_summary(shopping_list: pd.DataFrame, monthly_budget: float) -> dict:
    """Returns a summary dict for display."""
    if shopping_list.empty:
        return {}
    total_spent = shopping_list["total_price"].sum()
    return {
        "total_items":     len(shopping_list),
        "total_spent":     round(total_spent, 2),
        "budget":          monthly_budget,
        "budget_used_pct": round(100 * total_spent / monthly_budget, 1),
        "remaining":       round(monthly_budget - total_spent, 2),
        "by_category":     shopping_list.groupby("category")["total_price"].sum().to_dict(),
        "mandatory_count": len(shopping_list[shopping_list["slot"] != "filler"]),
    }
