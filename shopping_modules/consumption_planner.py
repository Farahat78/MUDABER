"""
consumption_planner.py — Monthly Household Needs Calculator
══════════════════════════════════════════════════════════════════════════════
Converts household info into realistic monthly consumption targets,
then maps those targets to available dataset products.

Science basis (Egyptian dietary patterns + WHO guidelines):
  Protein:    150–200g raw meat equiv per person/day
  Vegetables: 300–400g per person/day
  Grains:     200–300g cooked (80–120g dry) per person/day
  Dairy:      300–400ml milk equiv per person/day
  Fruits:     200–300g per person/day
  Fats/Oils:  20–30g per person/day

PUBLIC API:
  compute_targets(family_size, days) → dict[str, ConsumptionTarget]
  find_optimal_packages(target, dataset, budget_cap) → list[PackageChoice]
  build_base_list(family_size, days, budget, dataset) → pd.DataFrame
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from fuzzy_matcher import norm

# ── Daily consumption per person (grams or mL, unless noted) ─────────────────
# These are conservative Egyptian household averages.
# "consumption" = grams of the raw/packaged product per person per day.
DAILY_PER_PERSON: dict[str, dict] = {
    # ── Proteins ─────────────────────────────────────────────────────────────
    "chicken":     {"g": 100, "category": "Proteins",  "keywords": ["دجاج","فراخ","chicken"]},
    "meat":        {"g": 60,  "category": "Proteins",  "keywords": ["لحم","بيف","beef","meat"]},
    "fish":        {"g": 50,  "category": "Proteins",  "keywords": ["سمك","تونا","fish"]},
    "eggs":        {"pcs": 1, "category": "Proteins",  "keywords": ["بيض","egg"]},
    # ── Dairy ────────────────────────────────────────────────────────────────
    "milk":        {"ml": 250,"category": "Dairy",     "keywords": ["حليب","لبن","milk"]},
    "yogurt":      {"g": 100, "category": "Dairy",     "keywords": ["زبادي","yogurt"]},
    "cheese":      {"g": 30,  "category": "Dairy",     "keywords": ["جبنة","جبن","cheese"]},
    # ── Vegetables ───────────────────────────────────────────────────────────
    "vegetables":  {"g": 300, "category": "Vegetables","keywords": ["خضار","خضروات","vegetables"]},
    # ── Fruits ───────────────────────────────────────────────────────────────
    "fruits":      {"g": 200, "category": "Fruits",    "keywords": ["فاكهة","فواكه","fruit"]},
    # ── Grains ───────────────────────────────────────────────────────────────
    "rice":        {"g": 80,  "category": "Grains",    "keywords": ["أرز","ارز","rice"]},
    "pasta":       {"g": 60,  "category": "Grains",    "keywords": ["مكرونة","باستا","pasta"]},
    "bread":       {"g": 100, "category": "Grains",    "keywords": ["خبز","توست","bread"]},
    "lentils":     {"g": 30,  "category": "Grains",    "keywords": ["عدس","فول","lentil"]},
    # ── Oils & Fats ──────────────────────────────────────────────────────────
    "oil":         {"ml": 25, "category": "Oils & Fats","keywords": ["زيت","oil"]},
    # ── Beverages ────────────────────────────────────────────────────────────
    "tea":         {"g": 5,   "category": "Beverages", "keywords": ["شاي","tea"]},
    # ── Cleaning (weekly estimate, not daily) ─────────────────────────────────
    "detergent":   {"g_week": 200, "category": "Cleaning & Personal Care",
                    "keywords": ["منظف","مسحوق","detergent"]},
    "shampoo":     {"ml_week": 50, "category": "Cleaning & Personal Care",
                    "keywords": ["شامبو","shampoo"]},
    "soap":        {"g_week": 75,  "category": "Cleaning & Personal Care",
                    "keywords": ["صابون","soap"]},
    "tissues":     {"packs_month": 4, "category": "Cleaning & Personal Care",
                    "keywords": ["مناديل","tissue"]},
}

# ── Package size extractor ─────────────────────────────────────────────────────
_SIZE_PATTERNS = [
    (r"(\d+(?:\.\d+)?)\s*كجم",  "kg",   1000),   # kilograms → grams
    (r"(\d+(?:\.\d+)?)\s*kg",   "kg",   1000),
    (r"(\d+(?:\.\d+)?)\s*جم",   "g",    1),
    (r"(\d+(?:\.\d+)?)\s*جرام", "g",    1),
    (r"(\d+(?:\.\d+)?)\s*gram", "g",    1),
    (r"(\d+(?:\.\d+)?)\s*g\b",  "g",    1),
    (r"(\d+(?:\.\d+)?)\s*لتر",  "L",    1000),   # litres → mL
    (r"(\d+(?:\.\d+)?)\s*lit",  "L",    1000),
    (r"(\d+(?:\.\d+)?)\s*مل",   "ml",   1),
    (r"(\d+(?:\.\d+)?)\s*ml",   "ml",   1),
    (r"-\s*(\d+)\s*(?:بيضة|بيضات|قطعة|قطع)", "pcs", 1),
]


def parse_package_size(product_name: str) -> tuple[float, str] | None:
    """
    Extracts (size_value, unit) from a product name.
    Returns None if no size found.
    unit is one of: 'g', 'ml', 'pcs', 'L'
    All 'kg' → converted to 'g', 'L' → 'ml' with multiplier.
    """
    for pattern, unit_label, multiplier in _SIZE_PATTERNS:
        m = re.search(pattern, product_name, re.IGNORECASE)
        if m:
            raw_val = float(m.group(1))
            if unit_label == "kg":
                return raw_val * multiplier, "g"
            if unit_label == "L":
                return raw_val * multiplier, "ml"
            return raw_val, unit_label
    return None


# ── Consumption target dataclass ──────────────────────────────────────────────
@dataclass
class ConsumptionTarget:
    """Monthly consumption target for one product category."""
    item:             str
    category:         str
    total_g:          float        = 0.0   # grams (or mL) needed for the month
    total_pcs:        int          = 0     # pieces needed (for eggs, packs, etc.)
    unit:             str          = "g"   # 'g' | 'ml' | 'pcs'
    daily_per_person: float        = 0.0
    keywords:         list[str]    = field(default_factory=list)


@dataclass
class PackageChoice:
    """One product chosen to fill (part of) a consumption target."""
    product_name:   str
    category:       str
    unit_size_g:    float          # size in grams or mL per package
    unit_size_label:str            # human-readable ("1 كجم", "500 جم", etc.)
    quantity:       int
    unit_price:     float
    total_price:    float
    covers_g:       float          # how much of the target this covers
    efficiency:     float          # price per 100g (lower = better)
    dataset_row:    pd.Series      = field(default_factory=pd.Series)


# ── Compute targets ────────────────────────────────────────────────────────────
def compute_targets(
    family_size: int,
    days:        int = 30,
) -> dict[str, ConsumptionTarget]:
    """
    Computes monthly consumption targets for all tracked items.
    Returns dict keyed by item name (e.g. "chicken", "rice").
    """
    targets: dict[str, ConsumptionTarget] = {}

    for item, data in DAILY_PER_PERSON.items():
        cat = data["category"]

        if "g" in data:
            total = data["g"] * family_size * days
            targets[item] = ConsumptionTarget(
                item=item, category=cat,
                total_g=total, unit="g",
                daily_per_person=data["g"],
                keywords=data["keywords"],
            )
        elif "ml" in data:
            total = data["ml"] * family_size * days
            targets[item] = ConsumptionTarget(
                item=item, category=cat,
                total_g=total, unit="ml",
                daily_per_person=data["ml"],
                keywords=data["keywords"],
            )
        elif "pcs" in data:
            total_pcs = int(data["pcs"] * family_size * days)
            targets[item] = ConsumptionTarget(
                item=item, category=cat,
                total_pcs=total_pcs, unit="pcs",
                daily_per_person=data["pcs"],
                keywords=data["keywords"],
            )
        elif "g_week" in data:
            total = data["g_week"] / 7 * family_size * days
            targets[item] = ConsumptionTarget(
                item=item, category=cat,
                total_g=total, unit="g",
                keywords=data["keywords"],
            )
        elif "ml_week" in data:
            total = data["ml_week"] / 7 * family_size * days
            targets[item] = ConsumptionTarget(
                item=item, category=cat,
                total_g=total, unit="ml",
                keywords=data["keywords"],
            )
        elif "packs_month" in data:
            targets[item] = ConsumptionTarget(
                item=item, category=cat,
                total_pcs=data["packs_month"],
                unit="pcs",
                keywords=data["keywords"],
            )

    return targets


# ── Find optimal packages for a target ────────────────────────────────────────
def find_optimal_packages(
    target:         ConsumptionTarget,
    dataset:        pd.DataFrame,
    budget_cap:     float = float("inf"),
    exclude_norms:  set   = None,
) -> list[PackageChoice]:
    """
    Finds the most efficient product(s) from the dataset to cover
    a consumption target.

    Strategy:
      1. Filter dataset to category + keyword match
      2. Parse package sizes
      3. Score by price-per-unit (efficiency)
      4. Determine optimal quantity = ceil(total_needed / package_size)
      5. Prefer LARGER packages (fewer units = simpler shopping)
      6. Return top 1–2 choices (for variety where appropriate)
    """
    if exclude_norms is None:
        exclude_norms = set()

    pool = dataset[dataset["category"] == target.category].copy()
    if pool.empty:
        return []

    # Keyword filter
    kw_mask = pd.Series([False] * len(pool), index=pool.index)
    for kw in target.keywords:
        kw_mask |= pool["product_name"].str.contains(
            norm(kw), case=False, na=False, regex=False)
    pool = pool[kw_mask]

    if pool.empty:
        pool = dataset[dataset["category"] == target.category].copy()

    # Exclude already-chosen norms
    if "product_name_norm" in pool.columns:
        pool = pool[~pool["product_name_norm"].isin(exclude_norms)]

    if pool.empty:
        return []

    # Parse sizes
    pool = pool.copy()
    pool["_parsed"] = pool["product_name"].apply(parse_package_size)
    pool = pool[pool["_parsed"].notna()].copy()

    if pool.empty:
        # No size info → fallback: pick cheapest, qty=1
        best = pool.nsmallest(1, "effective_price")
        if best.empty:
            best = dataset[dataset["category"] == target.category].nsmallest(1, "effective_price")
        if best.empty:
            return []
        row = best.iloc[0]
        return [PackageChoice(
            product_name=row["product_name"],
            category=row["category"],
            unit_size_g=100, unit_size_label="?",
            quantity=1,
            unit_price=row["effective_price"],
            total_price=row["effective_price"],
            covers_g=100, efficiency=row["effective_price"],
            dataset_row=row,
        )]

    pool[["_size_val", "_size_unit"]] = pd.DataFrame(
        pool["_parsed"].tolist(), index=pool.index)

    # Compute efficiency (price per 100 grams/mL)
    pool["_efficiency"] = pool["effective_price"] / (pool["_size_val"] / 100)

    # Determine needed total
    total_needed = (
        target.total_pcs if target.unit == "pcs" else target.total_g
    )
    if total_needed <= 0:
        total_needed = 1.0

    choices: list[PackageChoice] = []
    seen: set[str] = set()

    # ── Size optimization strategy ────────────────────────────────────────────
    # Sort: LARGEST first (prefer fewer, bigger units) then by efficiency
    pool_sorted = pool.sort_values(["_size_val", "_efficiency"],
                                   ascending=[False, True])

    for _, row in pool_sorted.iterrows():
        if row["product_name"] in seen:
            continue

        size_val  = row["_size_val"]
        size_unit = row["_size_unit"]
        price     = row["effective_price"]

        # Compute optimal quantity
        qty = max(1, -(-int(total_needed) // max(1, int(size_val))))  # ceiling div
        total_cost = price * qty

        # Budget cap per item
        if total_cost > budget_cap * 0.4:
            qty = max(1, int(budget_cap * 0.4 // price))
            total_cost = price * qty

        # Build human-readable size label
        if size_unit == "g":
            size_label = f"{int(size_val)}جم" if size_val < 1000 else f"{size_val/1000:.1f}كجم"
        elif size_unit == "ml":
            size_label = f"{int(size_val)}مل" if size_val < 1000 else f"{size_val/1000:.1f}لتر"
        else:
            size_label = f"{int(size_val)} قطعة"

        choices.append(PackageChoice(
            product_name    = row["product_name"],
            category        = row["category"],
            unit_size_g     = size_val,
            unit_size_label = size_label,
            quantity        = qty,
            unit_price      = price,
            total_price     = round(total_cost, 2),
            covers_g        = size_val * qty,
            efficiency      = round(row["_efficiency"], 2),
            dataset_row     = row,
        ))
        seen.add(row["product_name"])

        if len(choices) >= 3:
            break

    return choices


# ── Build complete base list ───────────────────────────────────────────────────
def build_base_list(
    family_size: int,
    days:        int,
    budget:      float,
    dataset:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Builds a consumption-aware monthly shopping list.
    Uses compute_targets + find_optimal_packages to determine
    correct quantities and preferred package sizes.

    Returns a DataFrame in the same schema as the generator output.
    """
    targets      = compute_targets(family_size, days)
    rows:   list[dict] = []
    used:   set[str]   = set()
    budget_left = budget

    # Allocate budget proportionally by category priority
    BUDGET_ALLOC = {
        "Proteins":                 0.28,
        "Dairy":                    0.12,
        "Vegetables":               0.16,
        "Fruits":                   0.08,
        "Grains":                   0.14,
        "Oils & Fats":              0.06,
        "Beverages":                0.06,
        "Cleaning & Personal Care": 0.07,
        "Snacks":                   0.02,
        "Spices & Sauces":          0.01,
    }

    for item, target in targets.items():
        cat_budget = budget * BUDGET_ALLOC.get(target.category, 0.03)
        choices    = find_optimal_packages(target, dataset, cat_budget, used)

        for choice in choices[:1]:   # one primary product per target
            row = {
                "category":          choice.category,
                "product_name":      choice.product_name,
                "product_name_norm": norm(choice.product_name),
                "unit_price":        choice.unit_price,
                "quantity":          choice.quantity,
                "total_price":       choice.total_price,
                "slot":              f"consumption_{item}",
                "source":            choice.dataset_row.get("source", "Carrefour Egypt"),
                "discount_pct":      choice.dataset_row.get("discount_pct", 0),
                "package_size":      choice.unit_size_label,
                "monthly_covers_g":  choice.covers_g,
                "target_item":       item,
            }
            rows.append(row)
            used.add(norm(choice.product_name))
            budget_left -= choice.total_price

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Size optimizer for existing list ─────────────────────────────────────────
def optimize_existing_sizes(
    shopping_list: pd.DataFrame,
    dataset:       pd.DataFrame,
    family_size:   int,
    days:          int = 30,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Reviews the current shopping list and suggests size upgrades.
    E.g.: 10 × 320g rice → 2 × 1kg rice (cheaper, simpler).

    Returns (updated_df, list of changes made).
    """
    targets = compute_targets(family_size, days)
    df      = shopping_list.copy()
    changes: list[dict] = []

    for item, target in targets.items():
        cat = target.category
        # Find rows in list belonging to this target's category + keywords
        kw_mask = pd.Series([False] * len(df), index=df.index)
        for kw in target.keywords:
            kw_mask |= df["product_name"].str.contains(norm(kw), case=False,
                                                        na=False, regex=False)
        cat_mask = df["category"] == cat
        target_rows = df[cat_mask & kw_mask]

        if target_rows.empty:
            continue

        # Check if current packaging is inefficient
        # (many small units when one big unit exists)
        total_qty    = int(target_rows["quantity"].sum())
        avg_price    = float(target_rows["unit_price"].mean())
        total_spend  = float(target_rows["total_price"].sum())

        # Look for a more efficient package in the dataset
        better = find_optimal_packages(
            target, dataset,
            budget_cap=total_spend * 1.1,   # allow slight budget increase for efficiency
            exclude_norms=set()
        )

        if not better:
            continue

        best = better[0]

        # Only upgrade if it reduces total units by ≥ 30% or saves ≥ 5%
        current_units = total_qty
        better_units  = best.quantity
        savings       = (total_spend - best.total_price) / total_spend if total_spend > 0 else 0

        if better_units < current_units * 0.7 or savings > 0.05:
            # Apply the upgrade
            first_idx = target_rows.index[0]
            df.at[first_idx, "product_name"]  = best.product_name
            df.at[first_idx, "unit_price"]    = best.unit_price
            df.at[first_idx, "quantity"]      = best.quantity
            df.at[first_idx, "total_price"]   = best.total_price
            if "package_size" in df.columns:
                df.at[first_idx, "package_size"] = best.unit_size_label

            # Remove extra rows of the same target
            if len(target_rows) > 1:
                extra_idx = target_rows.index[1:]
                df = df.drop(extra_idx).reset_index(drop=True)

            changes.append({
                "item":       item,
                "old":        f"{current_units} × {target_rows['product_name'].iloc[0][:40]}",
                "new":        f"{best.quantity} × {best.product_name[:40]}",
                "old_spend":  round(total_spend, 1),
                "new_spend":  round(best.total_price, 1),
                "saved":      round(total_spend - best.total_price, 1),
                "reason":     "حزم أكبر = أوفر وأبسط",
            })

    return df.reset_index(drop=True), changes
