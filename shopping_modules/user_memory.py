"""
user_memory.py — User Preference Memory Layer
══════════════════════════════════════════════════════════════════════════
Tracks user behaviour patterns across sessions and distils them into
actionable preference signals for the AI Decision Planner and Generator.

Tracked signals:
  • removed_items   → items consistently deleted → avoid in generation
  • added_items     → items consistently requested → prioritise
  • avoided_cats    → categories reduced 3+ times
  • preferred_cats  → categories increased 3+ times
  • price_preference → cheap / quality
  • lifestyle_modes  → diet / gym / baby

PUBLIC API:
  load_memory(session_id)                      → MemoryContext
  save_interaction(session_id, before, after, intent, modes)
  get_generation_hints(session_id)             → GenerationHints
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

MEMORY_DIR = "data"
_MEM_TEMPLATE = "data/user_memory_{sid}.json"
MAX_HISTORY = 150


# ── MemoryContext — distilled preference snapshot ─────────────────────────────
@dataclass
class MemoryContext:
    session_id: str
    avoided_items: list[str] = field(default_factory=list)
    preferred_items: list[str] = field(default_factory=list)
    avoided_categories: list[str] = field(default_factory=list)
    preferred_categories: list[str] = field(default_factory=list)
    prefer_cheap: bool = False
    prefer_quality: bool = False
    diet_mode: bool = False
    gym_mode: bool = False
    baby_mode: bool = False
    interaction_count: int = 0

    @property
    def has_preferences(self) -> bool:
        return bool(
            self.avoided_items or self.preferred_items
            or self.avoided_categories or self.preferred_categories
            or self.prefer_cheap or self.prefer_quality
            or self.diet_mode or self.gym_mode or self.baby_mode
        )

    def to_prompt_context(self) -> str:
        lines: list[str] = []
        if self.avoided_items:
            lines.append(
                "المستخدم يتجنب: " + "، ".join(self.avoided_items[:6])
            )
        if self.preferred_items:
            lines.append(
                "المستخدم يفضل: " + "، ".join(self.preferred_items[:6])
            )
        if self.avoided_categories:
            lines.append(
                "الفئات المتجنبة: " + "، ".join(self.avoided_categories)
            )
        if self.preferred_categories:
            lines.append(
                "الفئات المفضلة: " + "، ".join(self.preferred_categories)
            )
        if self.prefer_cheap:
            lines.append("المستخدم يفضل المنتجات الاقتصادية")
        if self.prefer_quality:
            lines.append("المستخدم يفضل المنتجات عالية الجودة")
        if self.diet_mode:
            lines.append("المستخدم يتبع نظام دايت أو أكل صحي")
        if self.gym_mode:
            lines.append("المستخدم يتردد على الجيم — يحتاج بروتين أعلى")
        if self.baby_mode:
            lines.append("الأسرة لديها طفل رضيع — يحتاج منتجات أطفال")
        if not lines:
            return ""
        return (
            f"[ذاكرة المستخدم — {self.interaction_count} تفاعل سابق]\n"
            + "\n".join(f"  • {l}" for l in lines)
        )

    def avoided_norms(self) -> set[str]:
        """Normalised product names to skip during generation."""
        from fuzzy_matcher import norm
        return {norm(n) for n in self.avoided_items}


# ── GenerationHints — hints for the shopping list generator ──────────────────
@dataclass
class GenerationHints:
    skip_norms: set[str] = field(default_factory=set)
    boost_categories: list[str] = field(default_factory=list)
    reduce_categories: list[str] = field(default_factory=list)
    prefer_cheap: bool = False
    prefer_quality: bool = False


# ── Persistence helpers ────────────────────────────────────────────────────────
def _mem_path(session_id: str) -> str:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    sid = session_id.replace("/", "_").replace("\\", "_")[:32]
    return _MEM_TEMPLATE.format(sid=sid)


def _load_raw(session_id: str) -> dict:
    path = _mem_path(session_id)
    if not os.path.exists(path):
        return {"interactions": [], "session_id": session_id}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"interactions": [], "session_id": session_id}


def _save_raw(session_id: str, data: dict) -> None:
    path = _mem_path(session_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ── Pattern analyser ───────────────────────────────────────────────────────────
def _analyse_patterns(interactions: list[dict]) -> dict:
    """Extracts patterns from interaction history."""
    removed_counter: Counter = Counter()
    added_counter: Counter = Counter()
    cat_decrease: Counter = Counter()
    cat_increase: Counter = Counter()
    cheap_count = 0
    quality_count = 0
    diet_count = 0
    gym_count = 0
    baby_count = 0

    for ix in interactions:
        for item in ix.get("removed", []):
            removed_counter[item] += 1
        for item in ix.get("added", []):
            added_counter[item] += 1
        for cat in ix.get("cats_decreased", []):
            cat_decrease[cat] += 1
        for cat in ix.get("cats_increased", []):
            cat_increase[cat] += 1

        intent = ix.get("intent", "").lower()
        modes = ix.get("modes", {})
        if modes.get("diet") or "صحي" in intent or "diet" in intent:
            diet_count += 1
        if modes.get("gym") or "gym" in intent or "جيم" in intent:
            gym_count += 1
        if modes.get("baby") or "baby" in intent or "طفل" in intent:
            baby_count += 1
        if "cheap" in intent or "رخيص" in intent:
            cheap_count += 1
        if "quality" in intent or "جودة" in intent:
            quality_count += 1

    n = max(1, len(interactions))
    return {
        "avoided_items":      [k for k, v in removed_counter.items() if v >= 2],
        "preferred_items":    [k for k, v in added_counter.items()   if v >= 2],
        "avoided_cats":       [k for k, v in cat_decrease.items()    if v >= 3],
        "preferred_cats":     [k for k, v in cat_increase.items()    if v >= 3],
        "prefer_cheap":       cheap_count >= 2,
        "prefer_quality":     quality_count >= 2,
        "diet_mode":          diet_count / n >= 0.5,
        "gym_mode":           gym_count / n >= 0.5,
        "baby_mode":          baby_count / n >= 0.3,
    }


# ── Public API ─────────────────────────────────────────────────────────────────
def load_memory(session_id: str) -> MemoryContext:
    """Loads and analyses user memory for a session."""
    raw = _load_raw(session_id)
    interactions = raw.get("interactions", [])
    if not interactions:
        return MemoryContext(session_id=session_id)

    p = _analyse_patterns(interactions)
    return MemoryContext(
        session_id          = session_id,
        avoided_items       = p["avoided_items"],
        preferred_items     = p["preferred_items"],
        avoided_categories  = p["avoided_cats"],
        preferred_categories= p["preferred_cats"],
        prefer_cheap        = p["prefer_cheap"],
        prefer_quality      = p["prefer_quality"],
        diet_mode           = p["diet_mode"],
        gym_mode            = p["gym_mode"],
        baby_mode           = p["baby_mode"],
        interaction_count   = len(interactions),
    )


def save_interaction(
    session_id:  str,
    before_list: pd.DataFrame,
    after_list:  pd.DataFrame,
    instruction: str,
    intent:      str = "",
    modes:       dict | None = None,
) -> None:
    """Records one user interaction and persists it."""
    if modes is None:
        modes = {}

    before_names = set(before_list["product_name"].tolist()) if not before_list.empty else set()
    after_names  = set(after_list["product_name"].tolist())  if not after_list.empty  else set()

    removed = list(before_names - after_names)
    added   = list(after_names  - before_names)

    # Category-level changes
    def cat_sizes(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        return df.groupby("category").size().to_dict()

    before_cats = cat_sizes(before_list)
    after_cats  = cat_sizes(after_list)

    cats_decreased = [
        c for c in before_cats
        if after_cats.get(c, 0) < before_cats[c]
    ]
    cats_increased = [
        c for c in after_cats
        if after_cats[c] > before_cats.get(c, 0)
    ]

    record = {
        "ts":            datetime.now().isoformat(),
        "instruction":   instruction,
        "intent":        intent,
        "modes":         modes,
        "removed":       removed[:20],
        "added":         added[:20],
        "cats_decreased": cats_decreased,
        "cats_increased": cats_increased,
    }

    raw = _load_raw(session_id)
    interactions = raw.get("interactions", [])
    interactions.append(record)
    # Keep only recent history
    raw["interactions"] = interactions[-MAX_HISTORY:]
    _save_raw(session_id, raw)


def get_generation_hints(session_id: str) -> GenerationHints:
    """Returns actionable hints for the shopping list generator."""
    mem = load_memory(session_id)
    return GenerationHints(
        skip_norms        = mem.avoided_norms(),
        boost_categories  = mem.preferred_categories,
        reduce_categories = mem.avoided_categories,
        prefer_cheap      = mem.prefer_cheap,
        prefer_quality    = mem.prefer_quality,
    )


def clear_memory(session_id: str) -> None:
    """Clears all stored memory for a session."""
    path = _mem_path(session_id)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
