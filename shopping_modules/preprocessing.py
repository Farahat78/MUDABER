"""
preprocessing.py
─────────────────────────────────────────────────────────────
Cleans the raw Carrefour Egypt CSV, normalises Arabic text,
maps Carrefour sub-categories → our 10 household categories,
removes duplicates, filters outliers, and exports a clean
DataFrame ready for the generator.
"""

import re
import unicodedata
import pandas as pd
import numpy as np

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

# ─── Sub-category → canonical category mapping ───────────────────────────────
SUBCATEGORY_MAP = {
    # Proteins
    "اللحوم والدواجن":             "Proteins",
    "اللحوم و الفراخ المجمدة":    "Proteins",
    "السمك والأطعمة البحرية":      "Proteins",
    "سمك ومأكولات بحرية":          "Proteins",
    "بيض":                         "Proteins",
    # Dairy
    "جبنة و لبنة":                 "Dairy",
    "الزبادي":                     "Dairy",
    "حليب ولبن":                   "Dairy",
    "كريمة طعام":                  "Dairy",
    "الأطعمة المبردة":             "Dairy",   # cold cuts / processed meats → still dairy aisle
    # Vegetables
    "خضروات":                      "Vegetables",
    "خضار وفواكه عضوية":           "Vegetables",
    "الفاكهة و الخضار المجمد":    "Vegetables",
    # Fruits
    "الفاكهة":                     "Fruits",
    "المكسرات والتمور والفواكه المجففة": "Fruits",
    # Grains
    "أرز , مكرونة والبقوليات":    "Grains",
    "السكر و مستلزمات الخبز":     "Grains",
    "خبز عربي، راب وغيرها":       "Grains",
    "خبز وغيره":                   "Grains",
    "منتجات الفطور الغذائية":     "Grains",
    "كرواسان، باستري وكيك":       "Grains",
    "مأكولات جاهزة":               "Grains",
    # Oils & Fats
    "الزبدة و السمن":              "Oils & Fats",
    "مكونات الطبخ":                "Oils & Fats",
    # Beverages
    "شاي":                         "Beverages",
    "قهوة":                        "Beverages",
    "العصائر":                     "Beverages",
    "مشروبات غازية":               "Beverages",
    "ماء":                         "Beverages",
    "مشروبات بودرة":               "Beverages",
    "أعشاب":                       "Beverages",
    "مشروبات الأطفال":             "Beverages",
    # Snacks
    "الشوكولاته والمعجنات":        "Snacks",
    "بسكويت، كراكرز وكيك":        "Snacks",
    "شيبس ومقبلات":                "Snacks",
    "الوجبات الجاهزة والمقبلات":  "Snacks",
    "مربي، عسل وغيرها":            "Snacks",
    "آيس كريم وحلويات":            "Snacks",
    "حلويات شرقية":                "Snacks",
    "بودينج و اكتر":               "Snacks",
    # Cleaning & Personal Care
    "مستلزمات التنظيف":            "Cleaning & Personal Care",
    "مساحيق غسيل وتنظيف":         "Cleaning & Personal Care",
    "مناديل ومواد للاستخدام لمرة واحدة": "Cleaning & Personal Care",
    "مناديل":                      "Cleaning & Personal Care",
    "مناديل المطبخ و رول تغليف الطعام": "Cleaning & Personal Care",
    "مناديل حمام و مناديل مطبخ":  "Cleaning & Personal Care",
    "مستلزمات الاستحمام وغيرها":  "Cleaning & Personal Care",
    "أكياس قمامة":                 "Cleaning & Personal Care",
    "معطر جو و شمع":               "Cleaning & Personal Care",
    "مبيدات حشرية":                "Cleaning & Personal Care",
    "مستلزمات الورق":              "Cleaning & Personal Care",
    # Spices & Sauces
    "توابل، صلصات و خل":          "Spices & Sauces",
}

# ─── Arabic text normalisation ───────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    """Normalises Arabic text: unify alef forms, remove tatweel, diacritics."""
    if not isinstance(text, str):
        return ""
    # Unify alef variants → bare alef
    text = re.sub(r"[أإآ]", "ا", text)
    # Unify yaa variants
    text = re.sub(r"ى", "ي", text)
    # Unify taa marbuta
    text = re.sub(r"ة", "ه", text)
    # Remove tatweel
    text = re.sub(r"ـ", "", text)
    # Remove harakat (diacritics U+064B → U+0652)
    text = re.sub(r"[\u064B-\u065F]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_price(val) -> float | None:
    """Converts price to float; returns None on failure."""
    if pd.isna(val):
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except ValueError:
        return None


def parse_discount(val) -> float:
    """Extracts numeric discount percentage; returns 0 if absent."""
    if pd.isna(val) or val == "":
        return 0.0
    if isinstance(val, (int, float)):
        return min(float(val), 100.0)
    match = re.search(r"(\d+(?:\.\d+)?)", str(val))
    if match:
        pct = float(match.group(1))
        return min(pct, 100.0)
    return 0.0


def map_category(row: pd.Series) -> str:
    """Maps a row to one of our 10 canonical categories."""
    sub = row["sub_category"]
    if sub in SUBCATEGORY_MAP:
        return SUBCATEGORY_MAP[sub]
    # Fallback by main_category
    mc = row["main_category"]
    fallback = {
        "Cleaning Tools":           "Cleaning & Personal Care",
        "Beverages":                "Beverages",
        "Dairy products & eggs":    "Dairy",
        "Frozen Foods":             "Proteins",
        "Fresh Foods":              "Proteins",
        "Vegetables & Fruits":      "Vegetables",
        "Bakery":                   "Grains",
        "Organic & Health Foods":   "Snacks",
        "Supermarket":              "Snacks",
    }
    return fallback.get(mc, "Snacks")


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes price outliers per category using IQR × 3 (very loose fence
    to keep legitimate premium products while discarding data-entry errors).
    """
    cleaned_parts = []
    for cat, group in df.groupby("category"):
        q1 = group["price"].quantile(0.05)
        q3 = group["price"].quantile(0.95)
        iqr = q3 - q1
        lo = max(0.5, q1 - 3 * iqr)
        hi = q3 + 3 * iqr
        cleaned_parts.append(group[(group["price"] >= lo) & (group["price"] <= hi)])
    return pd.concat(cleaned_parts).reset_index(drop=True)


def load_and_clean(path_or_df) -> pd.DataFrame:
    """
    Full 14-step cleaning pipeline:
    1  Load CSV or accept DataFrame
    2  Drop rows with null product names
    3  Convert price to float
    4  Drop rows with null / zero prices
    5  Parse discount percentage
    6  Compute effective (discounted) price
    7  Normalise Arabic product name
    8  Normalise Arabic sub-category
    9  Map to canonical category
    10 Drop categories we don't handle (School Supplies, Baby Products…)
    11 Drop strict duplicates (same name + price)
    12 Remove price outliers per category
    13 Add price_per_unit estimate
    14 Reset index + sort
    """
    # 1
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    # 2
    df = df.dropna(subset=["product_name"])

    # 3
    df["price"] = df["price"].apply(clean_price)

    # 4
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # 5
    df["discount_pct"] = df["discount"].apply(parse_discount)

    # 6 - The scraped price is already the final discounted price
    df["effective_price"] = df["price"]

    # 7
    df["product_name_norm"] = df["product_name"].apply(normalize_arabic)

    # 8
    df["sub_category_norm"] = df["sub_category"].apply(normalize_arabic)

    # 9
    df["category"] = df.apply(map_category, axis=1)

    # 10  — keep only our 10 categories (implicitly done by mapping;
    #       but drop rows mapped to unexpected strings just in case)
    df = df[df["category"].isin(CATEGORIES)]

    # 11
    df = df.drop_duplicates(subset=["product_name_norm", "effective_price"])

    # 12
    df = remove_outliers(df)

    # 13 — rough unit price (price / 1 assuming single unit; useful for future
    #      quantity optimisation)
    df["price_per_unit"] = df["effective_price"]

    # 14
    df = df.sort_values(["category", "effective_price"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_and_clean("data/carrefour_products.csv")
    print(f"Clean dataset: {len(df)} rows")
    print(df["category"].value_counts())
    df.to_csv("data/carrefour_clean.csv", index=False, encoding="utf-8-sig")
    print("Saved → data/carrefour_clean.csv")
