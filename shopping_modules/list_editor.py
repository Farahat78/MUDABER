"""
list_editor.py — Human-Like Shopping List Editor
══════════════════════════════════════════════════════════════════════════════
This module replaces the action-parsing approach.

Instead of:
  user text → parse actions → apply actions

We now do:
  user text + current list + available dataset
    → Gemini thinks like a human editor
    → returns updated_list directly + changes_made + reasoning

Why this is better:
  1. Gemini handles "مش عايز في الالبان غير لبن ابيض وجبنة بيضا" naturally
  2. Complex multi-step edits ("غير الفاكهة كلها") work without brittle parsing
  3. The model can reason about the whole list holistically
  4. Arabic dialect, implicit meanings, context-awareness all work

Safety layer (post-Gemini):
  - Verify all products in updated_list exist in dataset OR current list
  - Flag and replace hallucinated products with fuzzy-matched alternatives
  - Ensure list is never empty
  - Enforce budget

PUBLIC API:
  edit_list(instruction, shopping_list, dataset, budget, family_size, days)
    → EditResult(updated_list, changes_made, reasoning, validation_log)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from fuzzy_matcher import find_in_dataset, find_in_list, norm, LOW_CONF, CONFIDENT
from consumption_planner import compute_targets

# ── Config ─────────────────────────────────────────────────────────────────────
_HARDCODED_KEY = "AIzaSyBpA8S9vLU-VmNlVHSlVsgCTkAGA9bVvG0"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", _HARDCODED_KEY)
GEMINI_MODEL   = "gemini-2.0-flash"

HALLUCINATION_CONFIDENCE = 60   # fuzzy score threshold for product validation


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class EditResult:
    updated_list:   pd.DataFrame
    changes_made:   list[str]
    reasoning:      str
    validation_log: list[dict]  = field(default_factory=list)
    intent:         str         = ""
    strategy:       str         = ""
    used_fallback:  bool        = False

    @property
    def summary(self) -> str:
        lines = []
        if self.changes_made:
            lines += [f"  • {c}" for c in self.changes_made]
        if self.reasoning:
            lines.append(f"💭 {self.reasoning}")
        return "\n".join(lines)


# ── List formatter for Gemini prompt ─────────────────────────────────────────
def _format_list(df: pd.DataFrame) -> str:
    """Formats the shopping list as a readable JSON array for Gemini."""
    if df.empty:
        return "[]"
    items = []
    for _, r in df.iterrows():
        size = r.get("package_size", "")
        item = {
            "name":     r["product_name"],
            "category": r["category"],
            "qty":      int(r["quantity"]),
            "price":    float(r["unit_price"]),
        }
        if size and size != "?":
            item["size"] = size
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=2)


def _format_dataset(df: pd.DataFrame, shopping_list: pd.DataFrame,
                    max_per_cat: int = 12) -> str:
    """Formats available dataset products grouped by category."""
    existing = set(shopping_list["product_name"].str.lower().tolist()
                   if not shopping_list.empty else [])
    lines = []
    for cat in ["Proteins","Dairy","Vegetables","Fruits","Grains",
                "Oils & Fats","Beverages","Snacks","Cleaning & Personal Care"]:
        pool = df[df["category"] == cat]
        pool = pool[~pool["product_name"].str.lower().isin(existing)]
        if pool.empty:
            continue
        cheap  = pool.nsmallest(6, "effective_price")
        sample = pool.sample(min(6, len(pool)), random_state=42)
        combined = pd.concat([cheap, sample]).drop_duplicates("product_name").head(max_per_cat)
        items = [f'"{r["product_name"]}" ({r["effective_price"]:.0f}ج.م)'
                 for _, r in combined.iterrows()]
        lines.append(f"  [{cat}]: " + " | ".join(items))
    return "\n".join(lines)


# ── Gemini system prompt (human editor mode) ──────────────────────────────────
_SYSTEM_PROMPT = """
أنت محرر قائمة تسوق بشري ذكي للأسر المصرية.
You are a human-like shopping list editor for Egyptian families.

دورك: تعديل قائمة التسوق بناءً على طلب المستخدم بطريقة طبيعية وذكية.
Your role: Edit the shopping list based on the user's request in a natural, intelligent way.

ستحصل على:
1. طلب المستخدم (عربي/إنجليزي/مزيج/عامية)
2. القائمة الحالية بالتفاصيل
3. المنتجات المتاحة من كارفور (استخدم هذه فقط عند الإضافة)
4. الميزانية المتبقية

══════════════════════════════════════════════════════
OUTPUT — أرجع JSON هذا فقط بدون markdown:
══════════════════════════════════════════════════════
{
  "updated_list": [
    {
      "name": "<اسم المنتج الحرفي>",
      "category": "<الفئة>",
      "qty": <كمية صحيحة>,
      "price": <سعر الوحدة>,
      "reason": "<لماذا هذا المنتج موجود أو أُضيف>"
    }
  ],
  "changes_made": [
    "<وصف واضح لكل تغيير>"
  ],
  "reasoning": "<شرح قصير لقرارك كله>",
  "intent": "<keep_only_dairy|replace_fruits|remove_item|etc>",
  "strategy": "<keep_only|replace_all|add_variety|remove|rebalance|increase|decrease>"
}

══════════════════════════════════════════════════════
كيف تفكر مثل إنسان — HOW TO THINK LIKE A HUMAN:
══════════════════════════════════════════════════════

مثال 1: "مش عايز في الالبان غير لبن ابيض وجبنة بيضا"
  → انظر للبنود في الالبان (Dairy) في القائمة الحالية
  → احتفظ فقط بما يطابق "لبن أبيض" و"جبنة بيضاء"
  → احذف الباقي (حليب بالشوكولاتة، زبادي، إلخ)
  → إذا لم يوجد "لبن أبيض" في القائمة، ابحث في المتاحة وأضفه

مثال 2: "غير الفاكهة كلها"
  → احذف كل الفاكهة الحالية
  → أضف 2-3 أنواع فاكهة مختلفة من المتاحة
  → تأكد من التنوع (لا تضع نفس الفاكهة مرتين)

مثال 3: "مش عايز اجنحة دجاج"
  → ابحث عن أجنحة الدجاج في القائمة
  → احذفها
  → إذا أراد المستخدم بديل (سياق واضح): أضف نوع دجاج مختلف من المتاحة
  → إذا لم يوجد سياق للاستبدال: احذف فقط

مثال 4: "خلي البروتين كله دجاج بس"
  → احذف: اللحم، السمك، أي بروتين آخر ليس دجاجاً
  → احتفظ بمنتجات الدجاج الموجودة
  → الكميات الجديدة = مجموع الكميات القديمة / عدد منتجات الدجاج المتبقية

مثال 5: "بدل كل المشروبات بعصاير طبيعية"
  → احذف كل المشروبات الحالية
  → أضف 2-3 أنواع عصير من المتاحة

مثال 6: "عايز تشكيلة خضار مختلفة"
  → لا تحذف كل الخضار (هناك خضار أساسية)
  → أضف 2-3 أنواع خضار جديدة من المتاحة
  → أو استبدل الأقل تنوعاً منها

مثال 7: "زود السمك" (Increase)
  → ابحث عن السمك في القائمة
  → ارفع الكمية (qty) بشكل منطقي (×1.5 أو +1)
  → إذا أراد تنوع: أضف نوع سمك آخر من المتاحة

══════════════════════════════════════════════════════
قواعد حاسمة — CRITICAL RULES:
══════════════════════════════════════════════════════
1. updated_list = القائمة الكاملة بعد التعديل (ليس فقط المضاف/المحذوف)
2. كل بند في updated_list: اسمه مطابق حرفياً لما في القائمة الحالية أو المتاحة
3. لا تخترع أسماء منتجات — فقط من القائمة الحالية أو المتاحة
4. changes_made: وصف واضح بالعربية لكل تغيير (حذف/إضافة/تعديل كمية)
5. إذا طلب المستخدم تغيير فئة كاملة: افعل ذلك بالكامل
6. لا ترجع قائمة فارغة أبداً
7. الأسماء في updated_list يجب أن تكون من المصادر الثلاثة:
   أ. اسم موجود في القائمة الحالية (للاحتفاظ به)
   ب. اسم موجود في المتاحة (للإضافة)
   ج. لا تخترع شيئاً جديداً

مثال على updated_list صحيحة:
[
  {"name": "أجنحة دجاج", "category": "Proteins", "qty": 2, "price": 56, "reason": "kept from original"},
  {"name": "سمك بلطى - وسط", "category": "Proteins", "qty": 1, "price": 24, "reason": "added for variety"},
  ...كل المنتجات الأخرى التي لم تتغير...
]
"""


# ── Call Gemini ────────────────────────────────────────────────────────────────
def _call_gemini(user_msg: str) -> dict | None:
    if not GOOGLE_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model   = GEMINI_MODEL,
            contents= user_msg,
            config  = types.GenerateContentConfig(
                system_instruction = _SYSTEM_PROMPT,
                max_output_tokens  = 4000,
                temperature        = 0.05,
            ),
        )
        raw = re.sub(r"```(?:json)?|```", "", response.text.strip()).strip()
        return json.loads(raw)
    except Exception:
        return None


# ── Product validator (anti-hallucination) ────────────────────────────────────
def _validate_and_fix_product(
    item:          dict,
    current_list:  pd.DataFrame,
    dataset:       pd.DataFrame,
) -> tuple[dict | None, dict]:
    """
    Validates a single product in Gemini's updated_list.
    Returns (fixed_item_or_None, log_entry).

    Checks:
    1. Exact match in current list → keep as-is
    2. Exact match in dataset → use dataset price
    3. Fuzzy match (score ≥ 60) → use matched name
    4. No match → skip (hallucinated product)
    """
    name     = item.get("name", "")
    category = item.get("category", "")

    # 1. Exact match in current list (kept from before)
    if not current_list.empty:
        exact_cur = current_list[current_list["product_name"] == name]
        if not exact_cur.empty:
            row = exact_cur.iloc[0]
            return {
                "product_name":      row["product_name"],
                "category":          row["category"],
                "unit_price":        row["unit_price"],
                "quantity":          max(1, int(item.get("qty", row["quantity"]))),
                "total_price":       round(row["unit_price"] * max(1, int(item.get("qty", row["quantity"]))), 2),
                "slot":              row.get("slot", "kept"),
                "source":            row.get("source", "Carrefour Egypt"),
                "discount_pct":      row.get("discount_pct", 0),
                "package_size":      row.get("package_size", ""),
            }, {"name": name, "source": "current_list", "score": 100, "status": "ok"}

    # 2. Exact match in dataset
    exact_ds = dataset[dataset["product_name"] == name]
    if not exact_ds.empty:
        row = exact_ds.iloc[0]
        qty = max(1, int(item.get("qty", 1)))
        return {
            "product_name":      row["product_name"],
            "category":          row["category"],
            "unit_price":        row["effective_price"],
            "quantity":          qty,
            "total_price":       round(row["effective_price"] * qty, 2),
            "slot":              "ai_added",
            "source":            row.get("source", "Carrefour Egypt"),
            "discount_pct":      row.get("discount_pct", 0),
            "product_name_norm": row.get("product_name_norm", norm(row["product_name"])),
        }, {"name": name, "source": "dataset_exact", "score": 100, "status": "ok"}

    # 3. Fuzzy match
    cat_for_search = category if category else None
    match = find_in_dataset(name, dataset, category=cat_for_search)

    if match and match.score >= HALLUCINATION_CONFIDENCE:
        row = match.row
        qty = max(1, int(item.get("qty", 1)))
        return {
            "product_name":      row["product_name"],
            "category":          row["category"],
            "unit_price":        row["effective_price"],
            "quantity":          qty,
            "total_price":       round(row["effective_price"] * qty, 2),
            "slot":              "ai_added",
            "source":            row.get("source", "Carrefour Egypt"),
            "discount_pct":      row.get("discount_pct", 0),
            "product_name_norm": row.get("product_name_norm", norm(row["product_name"])),
        }, {"name": name, "matched_to": row["product_name"],
            "source": "fuzzy", "score": match.score, "status": "ok"}

    # Also try fuzzy in current list
    if not current_list.empty:
        hits = find_in_list(name, current_list)
        if hits and hits[0].score >= HALLUCINATION_CONFIDENCE:
            row = hits[0].row
            qty = max(1, int(item.get("qty", row.get("quantity", 1))))
            return {
                "product_name":      row["product_name"],
                "category":          row["category"],
                "unit_price":        row["unit_price"],
                "quantity":          qty,
                "total_price":       round(row["unit_price"] * qty, 2),
                "slot":              row.get("slot", "kept"),
                "source":            row.get("source", "Carrefour Egypt"),
                "discount_pct":      row.get("discount_pct", 0),
            }, {"name": name, "matched_to": row["product_name"],
                "source": "list_fuzzy", "score": hits[0].score, "status": "ok"}

    # 4. Hallucinated — skip
    return None, {"name": name, "source": "none", "score": 0,
                  "status": "hallucinated", "skipped": True}


def _build_updated_df(
    gemini_list:   list[dict],
    current_list:  pd.DataFrame,
    dataset:       pd.DataFrame,
    budget:        float,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Validates every item in Gemini's updated_list, fixes hallucinations,
    returns (final_df, validation_log).
    """
    rows: list[dict] = []
    validation_log: list[dict] = []
    used_names: set[str] = set()
    total_cost  = 0.0

    for item in gemini_list:
        fixed, log = _validate_and_fix_product(item, current_list, dataset)
        validation_log.append(log)

        if fixed is None:
            continue  # hallucinated, skip

        # Duplicate prevention
        pname = fixed["product_name"]
        if pname in used_names:
            log["status"] = "duplicate_skipped"
            continue

        # Budget guard
        if total_cost + fixed["total_price"] > budget * 1.05:
            log["status"] = "budget_exceeded"
            continue

        rows.append(fixed)
        used_names.add(pname)
        total_cost += fixed["total_price"]

    if not rows:
        # Safety: return original list if everything was rejected
        return current_list.copy(), validation_log

    df = pd.DataFrame(rows)
    if "product_name_norm" not in df.columns:
        df["product_name_norm"] = df["product_name"].apply(norm)
    return df.reset_index(drop=True), validation_log


# ── Rule-based fallback (no Gemini) ───────────────────────────────────────────
def _rule_based_edit(
    instruction:  str,
    shopping_list: pd.DataFrame,
    dataset:       pd.DataFrame,
    budget:        float,
) -> EditResult:
    """
    Handles the hardest cases without Gemini using smart rule logic.
    Triggered when Gemini is unavailable or returns invalid output.
    """
    from smart_modify import _pick_new_items, _ensure_norm, _used_norms
    from fuzzy_matcher import CATEGORY_SYNONYMS, CONCEPTS

    text   = norm(instruction)
    df     = _ensure_norm(shopping_list.copy())
    changes: list[str] = []

    # ── Pattern: "مش عايز في X غير A وB" (keep only A, B in category X) ─────
    keep_only_patterns = [
        r'(?:مش عايز|ما أريد|بس|فقط).+(?:غير|إلا|except|only)',
        r'(?:احتفظ|keep).+(?:فقط|only|بس)',
        r'(?:i want|I want).+only.+in',   # "I want only X in Y"
        r'only.+in.+(?:proteins?|dairy|fruits?|grains?|snacks?)',
    ]
    is_keep_only = any(re.search(p, text) for p in keep_only_patterns) or \
                   any(norm(k) in text for k in ["مش عايز غير","مش عايز في","keep only","احتفظ"])

    # ── Pattern: "غير X كلها" / "بدل X كلها" (replace entire category) ──────
    replace_all_patterns = [
        r'(?:غير|بدل|change|replace).+(?:كلها|كله|كل|all)',
        r'(?:تشكيلة|variety|تنوع|مختلف).+(?:خضار|فاكهة|بروتين)',  # "تشكيلة خضار مختلفة"
        r'(?:خضار|فاكهة|بروتين).+(?:مختلف|تشكيلة|variety)',
    ]
    is_replace_all = any(re.search(p, text) for p in replace_all_patterns)

    # ── Smarter category detection ─────────────────────────────────────────
    # Uses multi-word phrases first, then single words
    # This prevents "بيض" in "الالبان" from matching Proteins
    CAT_PHRASES: list[tuple[str, str]] = [
        ("الالبان", "Dairy"), ("الأالبان", "Dairy"), ("ألبان", "Dairy"),
        ("البروتين", "Proteins"), ("البروتينات", "Proteins"),
        ("الخضار", "Vegetables"), ("الخضروات", "Vegetables"),
        ("الفاكهة", "Fruits"), ("الفواكه", "Fruits"),
        ("النشويات", "Grains"), ("الكارب", "Grains"),
        ("المشروبات", "Beverages"), ("المشروب", "Beverages"),
        ("السناكس", "Snacks"), ("الحلويات", "Snacks"),
        ("التنظيف", "Cleaning & Personal Care"),
    ]

    target_cat: str | None = None
    # Try phrases first
    for phrase, cat_en in CAT_PHRASES:
        if norm(phrase) in text:
            target_cat = cat_en
            break
    # Fallback to single-word synonyms (but skip very short ones that cause false positives)
    if not target_cat:
        for cat_ar, cat_en in CATEGORY_SYNONYMS.items():
            if len(cat_ar) >= 3 and norm(cat_ar) in text:
                # Avoid matching "بيض" inside longer words like "بيضا"
                cat_norm = norm(cat_ar)
                idx = text.find(cat_norm)
                if idx >= 0:
                    before = text[idx-1] if idx > 0 else " "
                    after  = text[idx+len(cat_norm)] if idx+len(cat_norm) < len(text) else " "
                    # Word boundary check
                    if before in " \t\n\u0020" and after in " \t\n\u0020وأ":
                        target_cat = cat_en
                        break

    # ── Execute keep_only ────────────────────────────────────────────────────
    if is_keep_only and target_cat:
        after_kw  = re.split(r'غير|إلا|except|only', instruction, maxsplit=1)
        before_bos = re.split(r'بس|فقط', instruction, maxsplit=1)
        keep_text  = norm(after_kw[-1]) if len(after_kw) > 1 else norm(before_bos[0])
        SKIP = {norm(w) for w in ['البروتين','البروتينات','الالبان','الخضار','الفاكهة',
                                   'النشويات','المشروبات','كله','كلها','خلي','عايز',
                                   'مش','في','من','على','بس','فقط','غير']}
        keep_words = [w for w in keep_text.split() if len(w) > 2 and w not in SKIP]
        if not keep_words:
            for concept, syns in CONCEPTS.items():
                if any(norm(s) in text for s in syns):
                    keep_words.append(concept)
        cat_rows  = df[df['category'] == target_cat]
        to_keep, to_remove = [], []
        for _, row in cat_rows.iterrows():
            pname = row['product_name']
            pnorm = norm(pname)
            matched = any(kw in pnorm for kw in keep_words) or                       any(norm(s) in pnorm for kw in keep_words
                          for concept, syns in CONCEPTS.items()
                          for s in syns if kw in norm(s))
            (to_keep if matched else to_remove).append(pname)
        if not to_keep and keep_words:
            hits = find_in_list(' '.join(keep_words), cat_rows)
            if hits and hits[0].score >= 50:
                to_keep  = [hits[0].product_name]
                to_remove = [r['product_name'] for _,r in cat_rows.iterrows()
                             if r['product_name'] not in to_keep]
        for pname in to_remove:
            df = df[df['product_name'] != pname].reset_index(drop=True)
            changes.append(f"حُذف '{pname[:40]}' (خارج قائمة الاحتفاظ)")
        used = _used_norms(df)
        for kw in keep_words[:3]:
            hits = find_in_list(kw, df)
            if not hits or hits[0].score < 55:
                match = find_in_dataset(kw, dataset, category=target_cat)
                if match and match.score >= 55 and match.row['product_name'] not in df['product_name'].tolist():
                    row = match.row
                    new_row = {'category': row['category'], 'product_name': row['product_name'],
                               'product_name_norm': norm(row['product_name']),
                               'unit_price': row['effective_price'], 'quantity': 1,
                               'total_price': row['effective_price'], 'slot': 'keep_only_add',
                               'source': row.get('source',''), 'discount_pct': row.get('discount_pct',0)}
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    changes.append(f"أُضيف '{row['product_name'][:40]}' (مطلوب في القائمة)")
        reasoning = f"احتُفظ بـ [{', '.join((to_keep or keep_words)[:3])}] في {target_cat}"
        return EditResult(df, changes, reasoning, intent='keep_only', strategy='keep_only')
    # ── Execute replace_all ──────────────────────────────────────────────────
    if is_replace_all and target_cat:
        # Remove all current items in the category
        removed = df[df["category"] == target_cat]["product_name"].tolist()
        df = df[df["category"] != target_cat].reset_index(drop=True)
        for pname in removed:
            changes.append(f"حُذف '{pname[:40]}'")

        # Add 2-3 new items from dataset
        budget_left = max(0, budget - float(df["total_price"].sum()))
        used        = _used_norms(df)
        new_prods   = _pick_new_items(target_cat, dataset, used, budget_left / 3, 3)
        for prod in new_prods:
            row = {
                "category": prod["category"],
                "product_name": prod["product_name"],
                "product_name_norm": prod.get("product_name_norm", norm(prod["product_name"])),
                "unit_price": prod["effective_price"],
                "quantity": 1,
                "total_price": prod["effective_price"],
                "slot": "replaced",
                "source": prod.get("source",""),
                "discount_pct": prod.get("discount_pct",0),
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            changes.append(f"أُضيف '{prod['product_name'][:40]}' (بديل جديد)")

        reasoning = f"تم استبدال كل {target_cat} بمنتجات متنوعة جديدة"
        return EditResult(df, changes, reasoning, intent="replace_all", strategy="replace_all")

    # ── Default: route to existing action-based system ───────────────────────
    from nlp_engine import force_fallback_plan
    from decision_engine import execute_plan

    plan   = force_fallback_plan(instruction, shopping_list, dataset, budget)
    result = execute_plan(plan, shopping_list, dataset, budget)

    return EditResult(
        updated_list  = result.shopping_list,
        changes_made  = [l.message_ar for l in result.action_logs
                         if l.outcome == "applied" and l.message_ar],
        reasoning     = plan.analysis_reason or plan.intent,
        validation_log= [],
        intent        = plan.intent,
        strategy      = plan.strategy,
        used_fallback = True,
    )


# ── Main entry point ──────────────────────────────────────────────────────────
def edit_list(
    instruction:   str,
    shopping_list: pd.DataFrame,
    dataset:       pd.DataFrame,
    budget:        float       = float("inf"),
    family_size:   int         = 4,
    days:          int         = 30,
) -> EditResult:
    """
    Human-like list editor.

    1. Build a rich prompt with current list + available dataset
    2. Call Gemini → it returns updated_list + changes_made + reasoning
    3. Validate every product in updated_list (anti-hallucination)
    4. If Gemini fails → rule-based fallback

    Returns EditResult with the final updated DataFrame.
    """
    if shopping_list.empty:
        return EditResult(shopping_list, ["القائمة فارغة"], "no-op")

    # Build prompt
    spend       = float(shopping_list["total_price"].sum())
    budget_left = max(0, budget - spend)

    user_msg = (
        f"طلب المستخدم: {instruction}\n\n"
        f"الميزانية: {budget:.0f}ج.م | مُنفق: {spend:.0f}ج.م | متبقي: {budget_left:.0f}ج.م\n"
        f"أسرة: {family_size} أفراد | مدة: {days} يوم\n\n"
        f"القائمة الحالية:\n{_format_list(shopping_list)}\n\n"
        f"المنتجات المتاحة في كارفور (للإضافة فقط):\n"
        f"{_format_dataset(dataset, shopping_list)}"
    )

    # Call Gemini
    raw = _call_gemini(user_msg)

    # Validate Gemini output
    if raw and isinstance(raw.get("updated_list"), list) and len(raw["updated_list"]) >= 1:
        updated_df, vlog = _build_updated_df(
            raw["updated_list"], shopping_list, dataset, budget
        )

        # If validation rejected everything, use original + fallback
        if len(updated_df) < 3:
            return _rule_based_edit(instruction, shopping_list, dataset, budget)

        changes   = raw.get("changes_made", [])
        reasoning = raw.get("reasoning", "")

        # Enhance changes with validation corrections
        hallucinated = [l["name"] for l in vlog if l.get("status") == "hallucinated"]
        fuzzy_fixed  = [f'{l["name"]} → {l["matched_to"]}' for l in vlog
                        if l.get("source") == "fuzzy"]
        if hallucinated:
            changes.append(f"⚠️ تم تجاهل {len(hallucinated)} منتج غير موجود: {', '.join(hallucinated[:3])}")
        if fuzzy_fixed:
            changes.append(f"🔄 تطابق تقريبي: {', '.join(fuzzy_fixed[:2])}")

        return EditResult(
            updated_list   = updated_df,
            changes_made   = changes,
            reasoning      = reasoning,
            validation_log = vlog,
            intent         = raw.get("intent", ""),
            strategy       = raw.get("strategy", ""),
            used_fallback  = False,
        )

    # Gemini failed or returned bad data → rule-based fallback
    result = _rule_based_edit(instruction, shopping_list, dataset, budget)
    result.used_fallback = True
    return result
