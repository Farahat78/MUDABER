"""
feedback.py — Orchestrator with Memory Integration
Pipeline:
  User Input → NLP Layer → Memory Retrieval → AI Planner → Force Engine → Final List
"""
from __future__ import annotations
import json, os
from datetime import datetime

import pandas as pd

from nlp_engine       import plan_modification, force_fallback_plan, DecisionPlan
from decision_engine  import execute_plan, ExecutionResult
from scorer           import score_result, ScoreBreakdown
from fuzzy_matcher    import norm

from nlp_rules    import parse_user_input
from force_engine import apply_modifications as _fe_apply, _has_valid_price
from user_memory  import load_memory, save_interaction, MemoryContext

FEEDBACK_FILE = "data/user_feedback.json"


def _df_to_list(df):
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "name": r["product_name"],
            "category": r.get("category", ""),
            "unit_price": float(r.get("unit_price", 0)),
            "_row": r.to_dict(),
        })
    return rows


def _dataset_to_list(dataset):
    rows = []
    for _, r in dataset.iterrows():
        price = float(r.get("effective_price", r.get("unit_price", 0)) or 0)
        if price <= 0:
            continue
        rows.append({
            "name": r["product_name"],
            "category": r.get("category", ""),
            "unit_price": price,
            "source": r.get("source", "Carrefour Egypt"),
            "discount_pct": float(r.get("discount_pct", 0)),
            "product_name_norm": r.get("product_name_norm", norm(r["product_name"])),
        })
    return rows


def _list_to_df(items, original_df):
    orig_by_name = {r["product_name"]: r.to_dict() for _, r in original_df.iterrows()}
    rows = []
    for item in items:
        name = item.get("name", "")
        if not name:
            continue
        if name in orig_by_name:
            rows.append(orig_by_name[name])
            continue
        if "_row" in item:
            rows.append(item["_row"])
            continue
        price = float(item.get("unit_price", 0))
        if price <= 0:
            continue
        qty = int(item.get("quantity", 1))
        rows.append({
            "category": item.get("category", ""),
            "product_name": name,
            "product_name_norm": item.get("product_name_norm", norm(name)),
            "unit_price": price,
            "quantity": qty,
            "total_price": round(price * qty, 2),
            "slot": "nlp_added",
            "source": item.get("source", "Carrefour Egypt"),
            "discount_pct": float(item.get("discount_pct", 0)),
        })

    if not rows:
        return original_df.copy()
    df_out = pd.DataFrame(rows)
    for col in original_df.columns:
        if col not in df_out.columns:
            df_out[col] = None
    if "unit_price" in df_out.columns:
        df_out = df_out[df_out["unit_price"].apply(
            lambda x: x is not None and float(x) > 0
        )].copy()
    if "unit_price" in df_out.columns and "quantity" in df_out.columns:
        mask = df_out["total_price"].isna() | (df_out["total_price"] == 0)
        df_out.loc[mask, "total_price"] = (
            df_out.loc[mask, "unit_price"] * df_out.loc[mask, "quantity"]
        ).round(2)
    return df_out.reset_index(drop=True)


def nlp_modify(instruction, shopping_list, dataset, budget=float("inf")):
    rules = parse_user_input(instruction)
    if not rules:
        return shopping_list, [], []
    sl_list = _df_to_list(shopping_list)
    ds_list = _dataset_to_list(dataset)
    updated = _fe_apply(sl_list, rules, ds_list)
    orig_names    = {i["name"] for i in sl_list}
    updated_names = {i["name"] for i in updated}
    added   = [i["name"] for i in updated if i["name"] not in orig_names]
    removed = [n for n in orig_names if n not in updated_names]
    changes = ([f"🗑️ حُذف: {n}" for n in removed] + [f"➕ أُضيف: {n}" for n in added]) or ["✅ تم التعديل"]
    updated_df = _list_to_df(updated, shopping_list)
    return updated_df, rules, changes


def process_modification(
    instruction, shopping_list, dataset, budget=float("inf"),
    session_id="default", family_size=4, days=30, base_list=None,
):
    if shopping_list.empty:
        s = ScoreBreakdown(0, 0, 0, 0, 0, ["القائمة فارغة"])
        return shopping_list, "⚠️ القائمة فارغة.", s, {}

    original = shopping_list.copy()

    # ── Step 1: Load user preference memory ──────────────────────────────────
    memory: MemoryContext = load_memory(session_id)
    memory_ctx_str = memory.to_prompt_context()

    # ── Step 2: Deterministic NLP path ────────────────────────────────────────
    updated_nlp, rules, changes = nlp_modify(instruction, shopping_list, dataset, budget)
    list_changed = (
        bool(rules) and (
            len(updated_nlp) != len(shopping_list) or
            set(updated_nlp["product_name"].tolist()) != set(shopping_list["product_name"].tolist())
        )
    )

    if list_changed:
        score   = score_result(original, updated_nlp, [], budget)
        n_rules = len(rules)
        summary = (
            f"[📋 NLP] {n_rules} قاعدة\n" +
            "\n".join(changes[:8]) +
            f"\n\n📊 النتيجة: {score.total:.0f}/100 ({score.grade})"
        )
        plan_dict = {
            "intent": "nlp_rules",
            "strategy": rules[0].get("action", "") if rules else "",
            "rules": rules, "changes": changes,
            "_score": score.total, "_applied": len(changes), "_skipped": 0,
        }
        _save_simple(instruction, plan_dict, score, session_id)
        save_interaction(session_id, original, updated_nlp, instruction,
                         intent=plan_dict["intent"], modes={})
        return updated_nlp, summary, score, plan_dict

    # ── Step 3: AI Decision Planner with memory context ───────────────────────
    plan: DecisionPlan = plan_modification(
        instruction, shopping_list, dataset, budget,
        family_size=family_size, days=days, base_list=base_list,
        memory_context=memory_ctx_str,
    )

    result: ExecutionResult = execute_plan(
        plan, shopping_list, dataset, budget,
        family_size=family_size, days=days,
    )

    from nlp_engine import Action as LA
    legacy_actions = [
        LA(type=a.type.replace("_quantity", ""),
           target=a.product, target_ar=a.product,
           scope="product", confidence=a.confidence, raw=a.raw)
        for a in plan.actions
    ]
    score   = score_result(original, result.shopping_list, legacy_actions, budget)
    s_icons = {"quantity": "⬆️", "variety": "🆕", "hybrid": "⬆️🆕",
                "preference": "⭐", "remove": "🗑️", "rebalance": "⚖️"}
    s_icon  = s_icons.get(plan.strategy, "🔧")
    flags   = ("👶 " if plan.baby_mode else "") + \
              ("🥗 " if plan.diet_mode else "") + \
              ("💪 " if plan.gym_mode  else "")
    lines   = [
        f"🤖 [{plan.strategy.upper()}] {flags}{s_icon} {result.applied_count} تعديل",
        f"💬 {plan.intent.replace('_', ' ')}",
        f"📊 تنوع: {plan.diversity_score:.0%} | {plan.analysis_reason[:55]}",
        "",
    ]
    if memory.has_preferences and memory_ctx_str:
        lines.insert(1, f"🧠 ذاكرة: {memory.interaction_count} تفاعل سابق مُستخدم")
    for log in result.action_logs:
        if log.outcome == "applied" and log.message_ar:
            lines.append(log.message_ar)
    skipped_logs = [l for l in result.action_logs if l.outcome == "skipped"]
    if skipped_logs:
        lines += ["", f"⚠️ تجاوز {len(skipped_logs)} تعديل:"]
        for sl2 in skipped_logs[:3]:
            lines.append(f"  • {sl2.message_ar[:70]}")
    lines += ["", f"📊 النتيجة: {score.total:.0f}/100 ({score.grade})"]

    summary   = "\n".join(l for l in lines if l is not None)
    plan_dict = _plan_to_dict(plan, result)
    plan_dict["_score"]   = score.total
    plan_dict["_applied"] = result.applied_count
    plan_dict["_skipped"] = result.skipped_count

    _save(instruction, plan_dict, result, score, session_id)
    save_interaction(
        session_id, original, result.shopping_list, instruction,
        intent=plan.intent,
        modes={"diet": plan.diet_mode, "gym": plan.gym_mode, "baby": plan.baby_mode},
    )
    return result.shopping_list, summary, score, plan_dict


def apply_modification(instruction, shopping_list, dataset=None,
                        monthly_budget=float("inf"), session_id="default",
                        use_api_fallback=True):
    if dataset is None:
        from preprocessing import load_and_clean
        try:    dataset = load_and_clean("data/carrefour_products.csv")
        except: return shopping_list, "⚠️ لا يمكن تطبيق التعديل بدون بيانات."
    updated, summary, _, _ = process_modification(
        instruction, shopping_list, dataset, monthly_budget, session_id)
    return updated, summary


def edit_shopping_list(instruction, shopping_list, dataset, budget=float("inf"),
                        session_id="default", family_size=4, days=30):
    from list_editor import edit_list, EditResult
    if shopping_list.empty:
        return shopping_list, "⚠️ القائمة فارغة.", {}
    original = shopping_list.copy()
    result: EditResult = edit_list(instruction, shopping_list, dataset, budget, family_size, days)
    score = score_result(original, result.updated_list, [], budget)
    engine = "📋 Rule" if result.used_fallback else "🤖 Gemini"
    lines = [
        f"[{engine}] {result.strategy or 'edit'}",
        f"💭 {result.reasoning[:70]}" if result.reasoning else "",
        "",
    ]
    if result.changes_made:
        lines += [f"  • {c}" for c in result.changes_made[:8]]
    lines += ["", f"📊 النتيجة: {score.total:.0f}/100 ({score.grade})"]
    summary = "\n".join(l for l in lines if l is not None)
    info = {
        "intent": result.intent, "strategy": result.strategy,
        "reasoning": result.reasoning, "changes": result.changes_made,
        "used_fallback": result.used_fallback, "_score": score.total,
    }
    _save_simple(instruction, {"intent": result.intent, "strategy": result.strategy,
        "baby_mode": False, "diet_mode": False, "gym_mode": False,
        "actions": [], "action_logs": []}, score, session_id)
    return result.updated_list, summary, info


def _plan_to_dict(plan, result):
    return {
        "intent": plan.intent, "strategy": plan.strategy,
        "diversity_score": plan.diversity_score,
        "analysis_reason": plan.analysis_reason,
        "strategy_reason": plan.strategy_reason,
        "baby_mode": plan.baby_mode, "diet_mode": plan.diet_mode,
        "gym_mode": plan.gym_mode, "intent_confidence": plan.intent_confidence,
        "actions": [{"type": a.type, "product": a.product, "reason": a.reason,
                     "confidence": a.confidence, "qty": a.quantity, "amount": a.amount,
                     "replacement": a.replacement, "old_size": a.old_size,
                     "new_size": a.new_size} for a in plan.actions],
        "action_logs": [{"type": l.action_type, "target": l.target,
                         "outcome": l.outcome, "score": float(l.validation.score)}
                        for l in result.action_logs],
    }


def _safe_json(v):
    if hasattr(v, "item"): return v.item()
    if isinstance(v, list): return [_safe_json(i) for i in v]
    if isinstance(v, dict): return {k: _safe_json(val) for k, val in v.items()}
    return v


def _save_simple(instruction, plan_dict, score, session_id):
    os.makedirs("data", exist_ok=True)
    record = _safe_json({
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id, "instruction": instruction,
        "intent": plan_dict.get("intent", ""), "strategy": plan_dict.get("strategy", ""),
        "baby_mode": plan_dict.get("baby_mode", False),
        "diet_mode": plan_dict.get("diet_mode", False),
        "gym_mode": plan_dict.get("gym_mode", False),
        "applied_count": plan_dict.get("_applied", 0),
        "skipped_count": 0, "score": float(score.total),
        "score_grade": score.grade, "plan": plan_dict,
    })
    existing = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except: existing = []
    existing.append(record)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def _save(instruction, plan_dict, result, score, session_id):
    _save_simple(instruction, {
        **plan_dict, "_applied": result.applied_count, "_skipped": result.skipped_count,
    }, score, session_id)


def load_feedback_history(session_id="default"):
    if not os.path.exists(FEEDBACK_FILE): return []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return [r for r in json.load(f) if r.get("session_id") == session_id]
    except: return []


def get_user_preferences(session_id="default"):
    history = load_feedback_history(session_id)
    return {
        "avg_score": round(sum(r.get("score", 0) for r in history) / max(1, len(history)), 1),
        "total_modifications": len(history),
        "baby_mode_used": any(r.get("baby_mode") for r in history),
        "diet_mode_used": any(r.get("diet_mode") for r in history),
        "gym_mode_used":  any(r.get("gym_mode")  for r in history),
        "top_strategies": list({r.get("strategy", "") for r in history})[:5],
    }


def save_feedback(*a, **kw): pass
