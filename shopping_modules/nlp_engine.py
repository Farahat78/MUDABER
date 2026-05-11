"""
nlp_engine.py  v9 — Household-Aware Gemini Decision Planner
══════════════════════════════════════════════════════════════════════════════
V9 changes:
  1. Receives household info (people_count, duration_days)
  2. Receives consumption targets (monthly grams needed)
  3. Receives original base list for drift detection
  4. Gemini now reasons like a household planner, not a keyword parser
  5. Actions include "update_size" for package optimization
  6. Generates final_list in output for full list regeneration

NEW action types:
  add | remove | replace | increase_quantity | decrease_quantity
  update_size   ← new: swap small packages for large
  rebalance     ← new: restore category to consumption target

PUBLIC API:
  plan_modification(instruction, shopping_list, dataset, budget,
                    family_size, days, base_list) → DecisionPlan
  force_fallback_plan(...)   → DecisionPlan (no Gemini needed)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
_HARDCODED_KEY = "AIzaSyBpA8S9vLU-VmNlVHSlVsgCTkAGA9bVvG0"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", _HARDCODED_KEY)
GEMINI_MODEL   = "gemini-2.0-flash"

KNOWN_CATEGORIES = [
    "Proteins","Dairy","Vegetables","Fruits","Grains",
    "Oils & Fats","Beverages","Snacks",
    "Cleaning & Personal Care","Spices & Sauces",
]


# ── Action dataclass ──────────────────────────────────────────────────────────
@dataclass
class PlannedAction:
    type:        str        # add|remove|replace|increase_quantity|decrease_quantity|update_size|rebalance
    product:     str        # exact product name from dataset or list
    category:    str        = ""
    quantity:    int        = 1
    amount:      int        = 1
    replacement: str        = ""
    old_size:    str        = ""
    new_size:    str        = ""
    reason:      str        = ""
    confidence:  float      = 0.85
    raw:         dict       = field(default_factory=dict)


@dataclass
class DecisionPlan:
    intent:           str
    strategy:         str
    diversity_score:  float
    analysis_reason:  str
    actions:          list[PlannedAction] = field(default_factory=list)
    baby_mode:        bool  = False
    diet_mode:        bool  = False
    gym_mode:         bool  = False
    intent_confidence:float = 0.85
    strategy_reason:  str   = ""
    raw:              dict  = field(default_factory=dict)

    @property
    def is_empty(self):
        return (not self.actions and not self.baby_mode
                and not self.diet_mode and not self.gym_mode)


# ── Context builders ──────────────────────────────────────────────────────────
def _list_context(shopping_list: pd.DataFrame) -> str:
    if shopping_list.empty:
        return "القائمة فارغة."
    lines = ["القائمة الحالية:"]
    for cat in KNOWN_CATEGORIES:
        rows = shopping_list[shopping_list["category"] == cat]
        if rows.empty:
            continue
        items = []
        for _, r in rows.iterrows():
            size = r.get("package_size", "")
            size_str = f" [{size}]" if size and size != "?" else ""
            items.append(f'"{r["product_name"][:45]}"{size_str} ×{r["quantity"]} ({r["unit_price"]:.0f}ج.م)')
        div = min(1.0, len(rows) / 6)
        lines.append(f"  [{cat}] تنوع≈{div:.1f}: " + " | ".join(items[:6]))
    return "\n".join(lines)


def _dataset_context(shopping_list: pd.DataFrame, dataset: pd.DataFrame,
                     n: int = 8) -> str:
    if dataset is None or dataset.empty:
        return "لا توجد منتجات في قاعدة البيانات."
    existing = set(shopping_list["product_name"].str.lower().tolist()
                   if not shopping_list.empty else [])
    lines = ["المنتجات المتاحة في كارفور (للإضافة أو الاستبدال):"]
    for cat in KNOWN_CATEGORIES:
        pool = dataset[dataset["category"] == cat]
        pool = pool[~pool["product_name"].str.lower().isin(existing)]
        if pool.empty:
            continue
        cheap  = pool.nsmallest(4, "effective_price")
        sample = pool.sample(min(4, len(pool)), random_state=1)
        combined = pd.concat([cheap, sample]).drop_duplicates("product_name").head(n)
        items = [f'"{r["product_name"][:45]}" ({r["effective_price"]:.0f}ج.م)'
                 for _, r in combined.iterrows()]
        lines.append(f"  [{cat}]: " + " | ".join(items))
    return "\n".join(lines)


def _targets_context(family_size: int, days: int) -> str:
    from consumption_planner import compute_targets
    targets = compute_targets(family_size, days)
    lines = [f"الاحتياجات الشهرية المحسوبة (أسرة {family_size} أفراد، {days} يوم):"]
    for item, t in list(targets.items())[:12]:
        if t.unit == "pcs":
            lines.append(f"  {item:15s}: {t.total_pcs} قطعة/شهر")
        else:
            kg = t.total_g / 1000
            lines.append(f"  {item:15s}: {t.total_g:.0f}{t.unit} = {kg:.1f}كجم/شهر")
    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """
أنت مخطط تسوق شهري ذكي للأسر المصرية.
You are a smart monthly shopping planner for Egyptian families.

ستحصل على:
1. طلب المستخدم  2. القائمة الحالية  3. الاحتياجات الشهرية المحسوبة
4. المنتجات المتاحة في كارفور  5. معلومات الأسرة والميزانية

دورك هو التخطيط كمخطط بشري — وليس زيادة الكميات بشكل أعمى.

══════════════════════════════════════
OUTPUT: أرجع JSON هذا فقط — بدون markdown:
══════════════════════════════════════
{
  "intent": "<increase_protein|remove_item|add_variety|rebalance_carbs|etc>",
  "strategy": "<quantity|variety|hybrid|preference|remove|rebalance>",
  "analysis": {
    "diversity_score": 0.0-1.0,
    "consumption_gap": "<ما ينقص أو يزيد>",
    "reason": "<لماذا هذه الاستراتيجية>"
  },
  "actions": [
    {
      "type": "add|remove|replace|increase_quantity|decrease_quantity|update_size|rebalance",
      "product": "<اسم من قائمة المتاحة أو الحالية>",
      "category": "<الفئة>",
      "quantity": 2,
      "amount": 1,
      "old_size": "<الحجم القديم>",
      "new_size": "<الحجم الجديد>",
      "replacement": "<اسم البديل>",
      "reason": "<سبب واضح>",
      "confidence": 0.0-1.0
    }
  ],
  "baby_mode": false,
  "diet_mode": false,
  "gym_mode": false,
  "reasoning": "<شرح قصير للقرار كله>",
  "debug": {
    "intent_confidence": 0.9,
    "strategy_reason": "<لماذا هذه الاستراتيجية>"
  }
}

══════════════════════════════════════
قواعد السلوك الذكي — SMART BEHAVIOR RULES:
══════════════════════════════════════
1. فكر في الاستهلاك الشهري الحقيقي — لا تزيد أو تنقص بشكل عشوائي
2. لا تضيف كميات صغيرة جداً — اختر الحزمة المناسبة
3. تحسين الحجم (update_size): إذا كانت 10×320جم أرز → اقترح 2×1كجم
4. التنوع الحقيقي — لا تكرر نفس النوع بشكل مختلف
5. لا تحذف فئة كاملة إلا بطلب صريح
6. الاحتياجات الشهرية هي مرجعك — ليس مجرد ما في القائمة
7. كن موضوعياً في diversity_score: 1=0.15، 2=0.30، 3-4=0.55، 5-6=0.70، 7+=0.85

══════════════════════════════════════
الاستراتيجيات:
══════════════════════════════════════
"quantity"   ← المستخدم طلب منتج محدد ("زود البيض")
"variety"    ← التنوع منخفض أو المستخدم طلب أنواع ("زود أنواع البروتين")
"hybrid"     ← طلب عام + تنوع متوسط ("زود بروتين")
"preference" ← تفضيل عام ("رخيص"، "صحي"، "دايت")
"remove"     ← حذف فقط
"rebalance"  ← إعادة ضبط القائمة على الاحتياجات الشهرية

══════════════════════════════════════
أمثلة:
══════════════════════════════════════
"زود بروتين وخليها متنوعة":
  {hybrid, add fish+eggs, increase existing chicken qty}

"زود البيض":
  {quantity, increase_quantity on existing egg product}

"قلل رز":
  {quantity, decrease_quantity on rice, possibly update_size to smaller}

"بدل أجنحة الدجاج بصدور":
  {hybrid, replace chicken wings→breast from AVAILABLE products}

"عايز أكل صحي":
  {preference, diet_mode=true, increase veg+fruit, decrease snacks}

"عندي طفل":
  {hybrid, baby_mode=true, add diapers+baby food}

"اعمل قائمة منطقية للشهر":
  {rebalance, check consumption targets, update_size for efficiency}

قواعد حاسمة:
1. product في add → من "المتاحة في كارفور" فقط
2. product في increase/decrease/remove → من "القائمة الحالية" فقط
3. أرجع JSON خالص فقط — لا أكواد markdown
4. confidence < 0.6 يعني الخوارزمية ستتجاهل الأمر
"""


# ── Gemini call ────────────────────────────────────────────────────────────────
def _call_gemini(user_msg: str) -> dict:
    if not GOOGLE_API_KEY:
        return {}
    try:
        client   = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model   = GEMINI_MODEL,
            contents= user_msg,
            config  = types.GenerateContentConfig(
                system_instruction = _SYSTEM_PROMPT,
                max_output_tokens  = 2500,
                temperature        = 0.05,
            ),
        )
        raw = re.sub(r"```(?:json)?|```", "", response.text.strip()).strip()
        return json.loads(raw)
    except Exception:
        return {}


def _build_plan(raw: dict) -> DecisionPlan:
    VALID = {"add","remove","replace","increase_quantity","decrease_quantity",
             "update_size","rebalance"}
    actions = []
    for a in raw.get("actions", []):
        if not isinstance(a, dict) or a.get("type") not in VALID:
            continue
        actions.append(PlannedAction(
            type        = a["type"],
            product     = a.get("product",""),
            category    = a.get("category",""),
            quantity    = max(1, int(a.get("quantity",1))),
            amount      = max(1, int(a.get("amount",1))),
            replacement = a.get("replacement",""),
            old_size    = a.get("old_size",""),
            new_size    = a.get("new_size",""),
            reason      = a.get("reason",""),
            confidence  = float(max(0.0, min(1.0, a.get("confidence",0.85)))),
            raw         = a,
        ))
    analysis = raw.get("analysis", {})
    debug    = raw.get("debug", {})
    return DecisionPlan(
        intent           = raw.get("intent","unknown"),
        strategy         = raw.get("strategy","hybrid"),
        diversity_score  = float(analysis.get("diversity_score", 0.5)),
        analysis_reason  = analysis.get("reason",""),
        actions          = actions,
        baby_mode        = bool(raw.get("baby_mode",False)),
        diet_mode        = bool(raw.get("diet_mode",False)),
        gym_mode         = bool(raw.get("gym_mode",False)),
        intent_confidence= float(debug.get("intent_confidence",0.85)),
        strategy_reason  = debug.get("strategy_reason",""),
        raw              = raw,
    )


# ── Main planner ───────────────────────────────────────────────────────────────
def plan_modification(
    instruction:    str,
    shopping_list:  pd.DataFrame,
    dataset:        pd.DataFrame,
    budget:         float        = float("inf"),
    family_size:    int          = 4,
    days:           int          = 30,
    base_list:      pd.DataFrame = None,
    memory_context: str          = "",
) -> DecisionPlan:
    """
    V9 planner — includes consumption targets + size context + memory context.
    Falls back to force_fallback_plan if Gemini returns nothing.
    """
    if not instruction.strip():
        return DecisionPlan("unknown","quantity",0.5,"",actions=[])

    spend       = float(shopping_list["total_price"].sum()) if not shopping_list.empty else 0
    budget_left = max(0.0, budget - spend)

    parts = [
        f"معلومات الأسرة: {family_size} أفراد | {days} يوم | ميزانية: {budget:.0f}ج.م | متبقي: {budget_left:.0f}ج.م",
        _targets_context(family_size, days),
        _list_context(shopping_list),
        _dataset_context(shopping_list, dataset),
    ]
    if memory_context:
        parts.append(memory_context)
    parts.append(f"طلب المستخدم: {instruction}")

    user_msg = "\n\n".join(parts)

    raw = _call_gemini(user_msg)
    if not raw or (not raw.get("actions") and not any([
        raw.get("baby_mode"), raw.get("diet_mode"), raw.get("gym_mode")])):
        return force_fallback_plan(instruction, shopping_list, dataset, budget,
                                   family_size, days)
    return _build_plan(raw)


# ── Force fallback ─────────────────────────────────────────────────────────────
def force_fallback_plan(
    instruction:  str,
    shopping_list:pd.DataFrame,
    dataset:      pd.DataFrame,
    budget:       float = float("inf"),
    family_size:  int   = 4,
    days:         int   = 30,
) -> DecisionPlan:
    """Pure-Python fallback. No Gemini required."""
    from fuzzy_matcher import find_in_list, find_in_dataset, norm, CATEGORY_SYNONYMS
    from smart_modify  import (decide_increase_mode, decide_decrease_mode,
                               _pick_new_items, _ensure_norm, _used_norms)
    from consumption_planner import compute_targets, find_optimal_packages

    text    = norm(instruction)
    actions: list[PlannedAction] = []

    REMOVE_KW   = ["مش عايز","شيل","احذف","بلاش","remove","delete"]
    INCREASE_KW = ["زود","زيد","اضف","ضيف","أضف","زيادة","more","increase","ضاعف"]
    DECREASE_KW = ["قلل","نقص","اقل","reduce","less","قليل"]
    REPLACE_KW  = ["بدل","استبدل","غير نوع","replace","swap"]
    BABY_KW     = ["طفل","بيبي","رضيع","baby","kids","حفاض"]
    HEALTHY_KW  = ["صحي","دايت","diet","healthy","رجيم"]
    CHEAP_KW    = ["رخيص","رخيصة","cheap","budget","وفر"]
    QUALITY_KW  = ["جودة","أحسن","premium","quality"]
    GYM_KW      = ["جيم","gym","عضلات","muscle"]
    SIZE_KW     = ["حجم أكبر","optimize","وفر","احسن حجم","حجم مناسب"]
    REBALANCE_KW= ["قائمة شهرية","اعمل قائمة","اعد","balance","منطقي"]

    def has(kws): return any(norm(k) in text for k in kws)

    baby_mode = has(BABY_KW)
    diet_mode = has(HEALTHY_KW)
    gym_mode  = has(GYM_KW)

    if has(HEALTHY_KW) or has(CHEAP_KW) or has(QUALITY_KW) or has(GYM_KW):
        pref = ("healthy" if has(HEALTHY_KW) else
                "cheap"   if has(CHEAP_KW)   else
                "quality" if has(QUALITY_KW) else "gym")
        return DecisionPlan(f"preference_{pref}","preference",0.6,
                            f"Global preference: {pref}",actions=[],
                            baby_mode=baby_mode,diet_mode=diet_mode,gym_mode=gym_mode,
                            intent_confidence=0.82)

    if baby_mode:
        return DecisionPlan("baby_mode","hybrid",0.5,"Baby products requested",
                            actions=[],baby_mode=True,intent_confidence=0.90)

    # Size optimization request
    if has(SIZE_KW) or has(REBALANCE_KW):
        return DecisionPlan("rebalance","rebalance",0.5,"Size and balance optimization",
                            actions=[],intent_confidence=0.80)

    words   = [w for w in instruction.split() if len(w) > 2]
    queries = [instruction] + words

    matched = None
    for q in queries:
        hits = find_in_list(q, shopping_list)
        if hits and hits[0].score >= 50:
            matched = hits[0]
            break

    target_cat = None
    for cat_ar, cat_en in CATEGORY_SYNONYMS.items():
        if norm(cat_ar) in text:
            target_cat = cat_en
            break

    # Remove
    if has(REMOVE_KW) and matched:
        actions.append(PlannedAction(type="remove",product=matched.product_name,
                                     category=matched.row["category"],
                                     reason="User requested removal",
                                     confidence=round(matched.score/100,2)))
        return DecisionPlan("remove_item","remove",0.6,"Remove request",
                            actions=actions,intent_confidence=0.85)

    # Replace
    KEEP_EXACT = ["مش عايز غير","بس ","فقط","keep only"]
    if has(REPLACE_KW) and not any(norm(k) in text for k in KEEP_EXACT) and matched:
        cat  = matched.row["category"]
        excl = _used_norms(_ensure_norm(shopping_list))
        repl = find_in_dataset(instruction, dataset, category=cat, exclude_norms=excl)
        if repl and repl.score >= 50:
            actions.append(PlannedAction(type="replace",product=matched.product_name,
                                         replacement=repl.row["product_name"],
                                         category=cat,reason="User replace request",
                                         confidence=round(min(matched.score,repl.score)/100,2)))
            return DecisionPlan("replace_item","hybrid",0.6,"Replace request",
                                actions=actions,intent_confidence=0.85)

    # Keep-only
    if any(norm(k) in text for k in KEEP_EXACT):
        keep_items = []
        for q in queries:
            hits = find_in_list(q, shopping_list)
            if hits and hits[0].score >= 50:
                pname = hits[0].product_name
                if pname not in keep_items:
                    keep_items.append(pname)
        if keep_items and not shopping_list.empty:
            try:
                cat = shopping_list[shopping_list["product_name"] == keep_items[0]]["category"].iloc[0]
                to_remove = shopping_list[(shopping_list["category"]==cat) &
                                         (~shopping_list["product_name"].isin(keep_items))]["product_name"].tolist()
                for pname in to_remove:
                    actions.append(PlannedAction(type="remove",product=pname,category=cat,
                                                 reason=f"Keep only: {keep_items}",confidence=0.82))
                return DecisionPlan("keep_only","remove",0.5,"Keep only request",
                                    actions=actions,intent_confidence=0.82)
            except Exception:
                pass

    # Increase
    if has(INCREASE_KW):
        cat = target_cat or (matched.row["category"] if matched else None)
        if matched and (not cat or matched.row["category"] == cat):
            # Specific product — quantity mode
            actions.append(PlannedAction(type="increase_quantity",
                                         product=matched.product_name,
                                         amount=max(1, int(matched.row.get("quantity",1))),
                                         reason="User increase request",
                                         confidence=round(matched.score/100,2)))
            return DecisionPlan(f"increase_{matched.product_name[:20]}","quantity",0.5,
                                "Specific product increase",actions=actions,intent_confidence=0.85)
        elif cat:
            decision = decide_increase_mode(shopping_list, cat, budget, hint_mode="auto")
            strategy = decision.mode
            if decision.items_to_add > 0:
                budget_left = max(0, budget - float(shopping_list["total_price"].sum()))
                price_cap   = budget_left / max(1, decision.items_to_add)
                new_prods   = _pick_new_items(cat, dataset, _used_norms(_ensure_norm(shopping_list)),
                                              price_cap, decision.items_to_add)
                for p in new_prods:
                    actions.append(PlannedAction(type="add",product=p["product_name"],
                                                 category=cat,quantity=1,
                                                 reason=f"Add variety to {cat}",confidence=0.80))
            if decision.qty_factor > 1.0:
                for _, row in shopping_list[shopping_list["category"]==cat].iterrows():
                    actions.append(PlannedAction(type="increase_quantity",
                                                 product=row["product_name"],
                                                 amount=max(1,round(row["quantity"]*(decision.qty_factor-1))),
                                                 reason=f"Quantity boost",confidence=0.78))
            if actions:
                return DecisionPlan(f"increase_{cat}",strategy,
                                    min(1.0,len(shopping_list[shopping_list["category"]==cat])/6),
                                    f"Increase {cat}",actions=actions,intent_confidence=0.82)

    # Decrease
    if has(DECREASE_KW):
        cat = target_cat or (matched.row["category"] if matched else None)
        if matched:
            actions.append(PlannedAction(type="decrease_quantity",product=matched.product_name,
                                         amount=max(1,round(matched.row.get("quantity",2)*0.4)),
                                         reason="Decrease request",
                                         confidence=round(matched.score/100,2)))
        elif cat:
            for _, row in shopping_list[shopping_list["category"]==cat].head(4).iterrows():
                actions.append(PlannedAction(type="decrease_quantity",product=row["product_name"],
                                             amount=max(1,round(row["quantity"]*0.4)),
                                             reason=f"Decrease {cat}",confidence=0.78))
        if actions:
            return DecisionPlan(f"decrease_{cat or 'item'}","quantity",0.5,
                                "Decrease request",actions=actions,intent_confidence=0.80)

    # Dataset search fallback
    for q in queries:
        m = find_in_dataset(q, dataset)
        if m and m.score >= 55:
            actions.append(PlannedAction(type="add",product=m.row["product_name"],
                                         category=m.row["category"],quantity=1,
                                         reason=f"Best match for '{q}'",
                                         confidence=round(m.score/100,2)))
            return DecisionPlan("add_item","variety",0.5,"Best-guess add",
                                actions=actions,intent_confidence=0.70)

    return DecisionPlan("unknown","quantity",0.5,"Could not determine intent",
                        actions=[],intent_confidence=0.0)


# ── Legacy compatibility shims ─────────────────────────────────────────────────
from dataclasses import dataclass as _dc, field as _f

@_dc
class Action:
    type:str="unknown"; scope:str="product"; target:str=""; target_ar:str=""
    replacement:str=""; replacement_ar:str=""; category:str=""
    amount:float=1.5; mode:str="auto"; allowed:list=_f(default_factory=list)
    allowed_ar:list=_f(default_factory=list); preference:str=""
    confidence:float=0.9; raw:dict=_f(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        return cls(type=d.get("type","unknown"), scope=d.get("scope","product"),
                   target=d.get("target",""), target_ar=d.get("target_ar",d.get("target","")),
                   replacement=d.get("replacement",""), replacement_ar=d.get("replacement_ar",""),
                   category=d.get("category",""), amount=float(d.get("amount",1.5)),
                   mode=d.get("mode","auto"), allowed=d.get("allowed",[]),
                   allowed_ar=d.get("allowed_ar",[]), preference=d.get("preference",""),
                   confidence=float(d.get("confidence",0.9)), raw=d)


def parse_intent(instruction, shopping_list, dataset=None):
    """Legacy shim."""
    if dataset is None:
        return {"actions":[],"baby_mode":False,"diet_mode":False,"gym_mode":False,"intent_summary":""}
    if isinstance(shopping_list, list):
        shopping_list = pd.DataFrame(shopping_list).rename(
            columns={"product":"product_name","price":"unit_price"})
    plan = plan_modification(instruction, shopping_list, dataset)
    legacy = []
    for a in plan.actions:
        if a.type == "add":
            legacy.append({"type":"add","target":a.product,"target_ar":a.product,
                           "category":a.category,"scope":"product","confidence":a.confidence})
        elif a.type == "increase_quantity":
            legacy.append({"type":"increase","target":a.product,"target_ar":a.product,
                           "scope":"product","amount":1.5,"mode":"quantity","confidence":a.confidence})
        elif a.type == "decrease_quantity":
            legacy.append({"type":"decrease","target":a.product,"target_ar":a.product,
                           "scope":"product","amount":0.6,"mode":"quantity","confidence":a.confidence})
        elif a.type == "remove":
            legacy.append({"type":"remove","target":a.product,"target_ar":a.product,
                           "scope":"product","confidence":a.confidence})
        elif a.type == "replace":
            legacy.append({"type":"replace","from":a.product,"from_ar":a.product,
                           "to":a.replacement,"to_ar":a.replacement,
                           "scope":"product","confidence":a.confidence})
    return {"actions":legacy,"baby_mode":plan.baby_mode,"diet_mode":plan.diet_mode,
            "gym_mode":plan.gym_mode,"intent_summary":plan.intent,
            "debug":{"strategy":plan.strategy,"fallback_used":True}}


def snapshot(sl):
    if isinstance(sl, pd.DataFrame):
        return (sl[["category","product_name","quantity","unit_price"]]
                .rename(columns={"product_name":"product","unit_price":"price"})
                .to_dict("records"))
    return sl

def build_data_context(sl, ds, n=8):
    return _dataset_context(sl, ds, n)
