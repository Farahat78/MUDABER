"""
fuzzy_matcher.py
════════════════════════════════════════════════════════════════════════
All fuzzy search logic lives here.

Three public functions:
  find_in_list(query, shopping_list)  → list of RowMatch (on current list)
  find_in_dataset(query, dataset, **filters) → best RowMatch (from Carrefour)
  norm(text) → normalized Arabic string

RowMatch carries: row (pd.Series), score (0–100), method, matched_term
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process as rfp

# ── Thresholds ────────────────────────────────────────────────────────────────
CONFIDENT   = 70   # ≥ 70  → apply directly
LOW_CONF    = 50   # 50–69 → apply with warning
# < 50 → rejected


# ── Arabic normalization ──────────────────────────────────────────────────────
def norm(text: str) -> str:
    """Unify alef/ya/ta-marbuta, strip harakat & tatweel, lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[\u064B-\u065F]", "", text)  # harakat
    return re.sub(r"\s+", " ", text).strip().lower()


# ── Concept map: canonical key → Arabic / English synonyms ───────────────────
CONCEPTS: dict[str, list[str]] = {
    "chicken":      ["فراخ","دجاج","فرخه","فرخة","دجاجه","صدر دجاج","اجنحه دجاج",
                     "صدور دجاج","ران دجاج","فراخ عادية","فراخ بلدي","فراخ كاملة",
                     "chicken","poultry"],
    "meat":         ["لحم","لحمه","بيف","لحم بقري","مكعبات لحم","كبده","beef","meat"],
    "fish":         ["سمك","بلطي","بوري","تونا","فيليه سمك","جمبري","fish","tuna","salmon"],
    "eggs":         ["بيض","بيضه","بيصه","egg","eggs"],
    "milk":         ["حليب","لبن","لبن طازج","لبن كامل","milk","full milk"],
    "white_cheese": ["جبنه بيضا","جبن ابيض","جبنة بيضاء","فيتا","white cheese"],
    "cheese":       ["جبنه","جبن","cheese","رومي","شيدر"],
    "yogurt":       ["زبادي","yogurt"],
    "tomato":       ["طماطم","تماتم","tomato"],
    "potato":       ["بطاطس","بطاطا","potato"],
    "onion":        ["بصل","بصله","onion"],
    "cucumber":     ["خيار","cucumber"],
    "carrot":       ["جزر","carrot"],
    "rice":         ["ارز","رز","rice"],
    "pasta":        ["مكرونه","باستا","سباجيتي","لسان عصفور","pasta","spaghetti"],
    "lentils":      ["عدس","عدس اصفر","فول","lentil","lentils"],
    "bread":        ["خبز","توست","عيش","bread","toast"],
    "tea":          ["شاي","tea"],
    "coffee":       ["قهوه","نسكافيه","قهوه سريعه","coffee","nescafe"],
    "juice":        ["عصير","juice"],
    "water":        ["ميه","مياه","water"],
    "oil":          ["زيت","زيت طبخ","زيت ذره","زيت نباتي","oil"],
    "butter":       ["زبده","سمن","butter","ghee"],
    "detergent":    ["منظف","مسحوق غسيل","سائل غسيل","detergent"],
    "shampoo":      ["شامبو","shampoo"],
    "soap":         ["صابون","جل استحمام","soap"],
    "tissues":      ["مناديل","ورق تواليت","tissue"],
    "chocolate":    ["شوكولاته","شيكولاته","chocolate"],
    "chips":        ["شيبس","كريسبي","chips"],
    "biscuits":     ["بسكويت","كراكر","biscuits"],
    "apple":        ["تفاح","تفاحه","apple"],
    "banana":       ["موز","موزه","banana"],
    "orange":       ["برتقال","برتقاله","orange"],
    "grapes":       ["عنب","grapes"],
    "mango":        ["مانجو","mango"],
    "watermelon":   ["بطيخ","watermelon"],
}

# Reverse: norm(synonym) → concept key
_REV: dict[str, str] = {
    norm(syn): key
    for key, syns in CONCEPTS.items()
    for syn in syns
}

# Category synonym → canonical name
CATEGORY_SYNONYMS: dict[str, str] = {
    "بروتين":"Proteins","فراخ":"Proteins","لحم":"Proteins","سمك":"Proteins",
    "بيض":"Proteins","proteins":"Proteins","protein":"Proteins","meat":"Proteins",
    "البان":"Dairy","ألبان":"Dairy","جبنه":"Dairy","زبادي":"Dairy",
    "حليب":"Dairy","dairy":"Dairy","لبن":"Dairy",
    "خضار":"Vegetables","خضروات":"Vegetables","vegetables":"Vegetables","veggies":"Vegetables",
    "فاكهه":"Fruits","فواكه":"Fruits","fruit":"Fruits","fruits":"Fruits",
    "نشويات":"Grains","ارز":"Grains","مكرونه":"Grains","خبز":"Grains",
    "grains":"Grains","carbs":"Grains","rice":"Grains","pasta":"Grains",
    "زيت":"Oils & Fats","سمن":"Oils & Fats","زبده":"Oils & Fats",
    "oils":"Oils & Fats","oil":"Oils & Fats","butter":"Oils & Fats",
    "مشروبات":"Beverages","شاي":"Beverages","قهوه":"Beverages","عصير":"Beverages",
    "beverages":"Beverages","drinks":"Beverages","tea":"Beverages","coffee":"Beverages",
    "سناكس":"Snacks","بسكويت":"Snacks","شيبس":"Snacks","حلويات":"Snacks",
    "شوكولاته":"Snacks","snacks":"Snacks","sweets":"Snacks","chips":"Snacks",
    "تنظيف":"Cleaning & Personal Care","منظف":"Cleaning & Personal Care",
    "شامبو":"Cleaning & Personal Care","صابون":"Cleaning & Personal Care",
    "مناديل":"Cleaning & Personal Care","cleaning":"Cleaning & Personal Care",
    "detergent":"Cleaning & Personal Care","soap":"Cleaning & Personal Care",
    "shampoo":"Cleaning & Personal Care","tissues":"Cleaning & Personal Care",
    "توابل":"Spices & Sauces","بهارات":"Spices & Sauces",
    "صلصه":"Spices & Sauces","spices":"Spices & Sauces","sauce":"Spices & Sauces",
}


# ── RowMatch ──────────────────────────────────────────────────────────────────
@dataclass
class RowMatch:
    row:          pd.Series
    score:        float          # 0–100
    method:       str
    matched_term: str

    @property
    def product_name(self) -> str:
        return self.row["product_name"]

    @property
    def confident(self) -> bool:
        return self.score >= CONFIDENT

    @property
    def low_conf(self) -> bool:
        return LOW_CONF <= self.score < CONFIDENT

    @property
    def rejected(self) -> bool:
        return self.score < LOW_CONF

    def to_log(self) -> dict:
        return {
            "product":  self.product_name,
            "score":    round(self.score, 1),
            "method":   self.method,
            "term":     self.matched_term,
            "level":    "HIGH" if self.confident else ("LOW" if self.low_conf else "REJECTED"),
        }


# ── Internal helpers ──────────────────────────────────────────────────────────
def _ensure_norm(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantees product_name_norm column exists (generator omits it)."""
    if "product_name_norm" not in df.columns:
        df = df.copy()
        df["product_name_norm"] = df["product_name"].apply(norm)
    return df


def _resolve_concept(query: str) -> list[str]:
    """Returns all synonyms for a query concept (including the query itself)."""
    q = norm(query)
    variants = [q, query.lower()]

    # Direct reverse lookup
    if q in _REV:
        concept_key = _REV[q]
        variants += [norm(s) for s in CONCEPTS[concept_key]]
        return list(dict.fromkeys(v for v in variants if v))

    # Partial match in reverse map
    for rev_norm, concept_key in _REV.items():
        if rev_norm in q or q in rev_norm:
            variants += [norm(s) for s in CONCEPTS[concept_key]]
            break

    # Check English concept keys
    for key in CONCEPTS:
        if key in query.lower():
            variants += [norm(s) for s in CONCEPTS[key]]

    return list(dict.fromkeys(v for v in variants if v))


def resolve_category(term: str) -> Optional[str]:
    """Maps a user term (Arabic or English) to canonical category name."""
    return CATEGORY_SYNONYMS.get(norm(term)) or CATEGORY_SYNONYMS.get(term.lower())


# ── Public: search in current shopping list ───────────────────────────────────
def find_in_list(
    query: str,
    shopping_list: pd.DataFrame,
    scope: str = "product",
    top_k: int = 5,
) -> list[RowMatch]:
    """
    Fuzzy-searches query against the CURRENT shopping list.
    Returns up to top_k RowMatch objects, sorted by score descending.
    scope: "product" | "category"
    """
    if shopping_list.empty or not query.strip():
        return []

    sl = _ensure_norm(shopping_list)

    # Category scope → return all rows in that category
    if scope == "category":
        cat = resolve_category(query)
        if cat:
            rows = sl[sl["category"] == cat]
            return [
                RowMatch(row=r, score=95.0, method="category_synonym", matched_term=query)
                for _, r in rows.iterrows()
            ]
        return []

    # Product scope: expand query to synonyms, run fuzzy on product_name_norm
    variants  = _resolve_concept(query)
    norms_col = sl["product_name_norm"].tolist()
    best: dict[str, RowMatch] = {}

    for variant in variants:
        if not variant:
            continue
        hits = rfp.extract(variant, norms_col, scorer=fuzz.partial_ratio,
                           limit=top_k, score_cutoff=35)
        for matched_norm, score, idx in hits:
            row   = sl.iloc[idx]
            pname = row["product_name"]
            if pname not in best or score > best[pname].score:
                best[pname] = RowMatch(
                    row=row, score=float(score),
                    method=f"fuzzy({variant})", matched_term=matched_norm,
                )

    return sorted(best.values(), key=lambda r: r.score, reverse=True)[:top_k]


# ── Public: search in full Carrefour dataset ──────────────────────────────────
def find_in_dataset(
    query: str,
    dataset: pd.DataFrame,
    category: Optional[str] = None,
    exclude_norms: Optional[set] = None,
    prefer_cheap: bool = False,
    prefer_quality: bool = False,
    max_price: float = float("inf"),
) -> Optional[RowMatch]:
    """
    Fuzzy-searches query in the FULL dataset to find a replacement product.
    Returns the single best RowMatch (>= LOW_CONF), or None.
    """
    pool = _ensure_norm(dataset.copy())

    if category:
        pool = pool[pool["category"] == category]
    if exclude_norms:
        pool = pool[~pool["product_name_norm"].isin(exclude_norms)]
    pool = pool[pool["effective_price"] <= max_price]

    if pool.empty:
        return None

    variants  = _resolve_concept(query)
    norms_col = pool["product_name_norm"].tolist()
    best: Optional[RowMatch] = None

    for variant in variants:
        if not variant:
            continue
        hits = rfp.extract(variant, norms_col, scorer=fuzz.partial_ratio,
                           limit=10, score_cutoff=35)
        for matched_norm, score, idx in hits:
            row = pool.iloc[idx]

            # Skip if price preference doesn't match
            from utils import PRICE_THRESHOLDS
            thresh = PRICE_THRESHOLDS.get(row["category"], {"cheap": 50, "quality": 150})
            if prefer_cheap and row["effective_price"] > thresh["cheap"]:
                continue
            if prefer_quality and row["effective_price"] < thresh["quality"]:
                continue

            if best is None or score > best.score:
                best = RowMatch(
                    row=row, score=float(score),
                    method=f"dataset_fuzzy({variant})", matched_term=matched_norm,
                )

    # If price filter killed everything, retry without
    if best is None and (prefer_cheap or prefer_quality):
        return find_in_dataset(query, dataset, category, exclude_norms,
                               prefer_cheap=False, prefer_quality=False,
                               max_price=max_price)

    return best if (best and best.score >= LOW_CONF) else None
