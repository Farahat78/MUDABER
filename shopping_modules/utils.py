"""
utils.py
─────────────────────────────────────────────────────────────
Shared utilities: budget allocation tables, quantity estimation,
composite scoring, and controlled-random selection helpers.
"""

import math
import random
import numpy as np
import pandas as pd

# ─── 10 canonical household categories ──────────────────────────────────────
CATEGORIES = [
    "Proteins",
    "Dairy",
    "Vegetables",
    "Fruits",
    "Grains",
    "Oils & Fats",
    "Beverages",
    "Snacks",
    "Cleaning & Personal Care",
    "Spices & Sauces",
]

# ─── Price tiers per category (Q25 = cheap, Q75 = quality) ──────────────────
PRICE_THRESHOLDS = {
    "Beverages":                {"cheap": 15,  "quality": 104},
    "Cleaning & Personal Care": {"cheap": 50,  "quality": 180},
    "Dairy":                    {"cheap": 35,  "quality": 109},
    "Fruits":                   {"cheap": 61,  "quality": 250},
    "Grains":                   {"cheap": 23,  "quality": 85},
    "Oils & Fats":              {"cheap": 31,  "quality": 155},
    "Proteins":                 {"cheap": 120, "quality": 285},
    "Snacks":                   {"cheap": 20,  "quality": 141},
    "Spices & Sauces":          {"cheap": 41,  "quality": 153},
    "Vegetables":               {"cheap": 24,  "quality": 62},
}

# ─── Budget allocation per lifestyle (must sum to 1.0) ──────────────────────
BUDGET_ALLOCATIONS = {
    "low": {
        "Proteins":                 0.25,
        "Vegetables":               0.20,
        "Grains":                   0.20,
        "Dairy":                    0.13,
        "Oils & Fats":              0.07,
        "Cleaning & Personal Care": 0.07,
        "Beverages":                0.04,
        "Spices & Sauces":          0.03,
        "Fruits":                   0.01,
        "Snacks":                   0.00,
    },
    "medium": {
        "Proteins":                 0.28,
        "Vegetables":               0.15,
        "Grains":                   0.14,
        "Dairy":                    0.12,
        "Oils & Fats":              0.08,
        "Cleaning & Personal Care": 0.08,
        "Beverages":                0.07,
        "Spices & Sauces":          0.03,
        "Fruits":                   0.03,
        "Snacks":                   0.02,
    },
    "high": {
        "Proteins":                 0.28,
        "Vegetables":               0.12,
        "Grains":                   0.11,
        "Dairy":                    0.12,
        "Oils & Fats":              0.07,
        "Cleaning & Personal Care": 0.08,
        "Beverages":                0.09,
        "Spices & Sauces":          0.03,
        "Fruits":                   0.05,
        "Snacks":                   0.05,
    },
}

# ─── Monthly quantity multipliers per family size ────────────────────────────
# These are realistic Egyptian household estimates (units or kg per month)
QUANTITY_BASE = {
    "Proteins": {
        "chicken":  {"unit": "kg",  "base": 4},
        "meat":     {"unit": "kg",  "base": 2},
        "fish":     {"unit": "kg",  "base": 2},
        "eggs":     {"unit": "pcs", "base": 30},
        "default":  {"unit": "pcs", "base": 2},
    },
    "Dairy": {
        "milk":     {"unit": "L",   "base": 4},
        "yogurt":   {"unit": "kg",  "base": 2},
        "cheese":   {"unit": "kg",  "base": 1},
        "default":  {"unit": "pcs", "base": 2},
    },
    "Vegetables":  {"default": {"unit": "kg",  "base": 3}},
    "Fruits":      {"default": {"unit": "kg",  "base": 2}},
    "Grains": {
        "rice":     {"unit": "kg",  "base": 3},
        "pasta":    {"unit": "kg",  "base": 2},
        "lentils":  {"unit": "kg",  "base": 1},
        "bread":    {"unit": "pcs", "base": 8},
        "default":  {"unit": "pcs", "base": 2},
    },
    "Oils & Fats": {"default": {"unit": "L",   "base": 2}},
    "Beverages":   {"default": {"unit": "pcs", "base": 3}},
    "Snacks":      {"default": {"unit": "pcs", "base": 2}},
    "Cleaning & Personal Care": {"default": {"unit": "pcs", "base": 1}},
    "Spices & Sauces":          {"default": {"unit": "pcs", "base": 1}},
}

# Family-size scaling factor (relative to family of 4)
def family_scale(family_size: int) -> float:
    return round(max(0.5, family_size / 4.0), 2)


def estimate_quantity(product_name: str, category: str, family_size: int) -> int:
    """
    Estimates monthly quantity for a product given family size.
    Uses keyword matching to find the right sub-type within a category.
    """
    scale = family_scale(family_size)
    cat_table = QUANTITY_BASE.get(category, {"default": {"unit": "pcs", "base": 1}})

    name_lower = product_name.lower()
    # Arabic keywords alongside English
    kw_map = {
        "chicken":  ["فراخ", "دجاج", "تركي", "chicken", "poultry"],
        "meat":     ["لحم", "beef", "meat", "بيف", "كبدة"],
        "fish":     ["سمك", "تونا", "fish", "salmon", "سردين"],
        "eggs":     ["بيض", "egg"],
        "milk":     ["حليب", "لبن", "milk"],
        "yogurt":   ["زبادي", "yogurt"],
        "cheese":   ["جبنة", "جبن", "cheese", "رومي", "فيتا"],
        "rice":     ["أرز", "ارز", "rice"],
        "pasta":    ["مكرونة", "pasta", "spaghetti", "شعيرية"],
        "lentils":  ["عدس", "lentils", "بقوليات"],
        "bread":    ["خبز", "توست", "bread"],
    }

    matched_type = "default"
    for sub_type, keywords in kw_map.items():
        if any(kw in name_lower for kw in keywords):
            matched_type = sub_type
            break

    info = cat_table.get(matched_type, cat_table.get("default", {"unit": "pcs", "base": 1}))
    qty = max(1, math.ceil(info["base"] * scale))
    return qty


# ─── Composite scoring ────────────────────────────────────────────────────────
def score_product(
    row: pd.Series,
    necessity_scores: dict[str, float],
    category_price_stats: dict[str, dict],
) -> float:
    """
    Composite score = 0.45 × necessity + 0.40 × price_efficiency + 0.15 × brand_diversity

    necessity        → lookup from NECESSITY_SCORES
    price_efficiency → (1 - (price - cat_min) / (cat_max - cat_min + ε)) normalised
    brand_diversity  → slight random jitter to prevent identical runs
    """
    cat = row["category"]
    price = row["effective_price"]

    necessity = necessity_scores.get(cat, 0.5)

    stats = category_price_stats.get(cat, {})
    cat_min = stats.get("min", 0)
    cat_max = stats.get("max", price + 1)
    price_eff = 1 - (price - cat_min) / (cat_max - cat_min + 1e-9)
    price_eff = float(np.clip(price_eff, 0, 1))

    diversity_jitter = random.uniform(0.0, 0.15)

    return 0.45 * necessity + 0.40 * price_eff + 0.15 * diversity_jitter


NECESSITY_SCORES = {
    "Proteins":                 1.0,
    "Vegetables":               0.95,
    "Grains":                   0.90,
    "Dairy":                    0.85,
    "Oils & Fats":              0.80,
    "Cleaning & Personal Care": 0.75,
    "Beverages":                0.65,
    "Fruits":                   0.60,
    "Spices & Sauces":          0.55,
    "Snacks":                   0.30,
}


def compute_price_stats(df: pd.DataFrame) -> dict[str, dict]:
    stats = {}
    for cat, group in df.groupby("category"):
        stats[cat] = {
            "min": group["effective_price"].min(),
            "max": group["effective_price"].max(),
            "mean": group["effective_price"].mean(),
        }
    return stats


def budget_for_category(
    total_budget: float,
    lifestyle: str,
    category: str,
) -> float:
    alloc = BUDGET_ALLOCATIONS.get(lifestyle, BUDGET_ALLOCATIONS["medium"])
    return total_budget * alloc.get(category, 0.02)


def format_currency(amount: float) -> str:
    return f"{amount:,.2f} ج.م"
