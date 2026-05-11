"""
pages/insights.py
────────────────────────────────────────────────────────────────────────────────
Smart Budget Insights page.

This page uses a LOCAL SQLite database (via insights_modules/database.py)
to store user expenses.  It does NOT read from pipeline CSV files — the user
enters expenses manually and the system tracks them.

IMPORTANT RULES:
  ✅ User enters expenses via the UI.
  ✅ Data is stored in a local SQLite DB (insights_modules/database.py).
  ✅ Smart insights are generated locally using insights_modules/ logic.
  ❌ Does NOT scrape any data.
  ❌ Does NOT run ML training.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_IM = os.path.join(os.path.dirname(__file__), "..", "insights_modules")
sys.path.insert(0, os.path.dirname(_IM))   # so "insights_modules" is importable
# Override the module name so existing imports work
import importlib
import importlib.util

def _load_insights_module(name: str, path: str):
    """Dynamically load a module from insights_modules/."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_IM, f"{name}.py"))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Make "modules.X" imports resolve to insights_modules too
    sys.modules[f"modules.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

# Bootstrap the insights modules so their internal `from modules import X` works
_modules_pkg = importlib.util.spec_from_file_location("modules", os.path.join(_IM, "__init__.py"))
_modules_mod = importlib.util.module_from_spec(_modules_pkg)
sys.modules["modules"] = _modules_mod
_modules_pkg.loader.exec_module(_modules_mod)

# Now import them normally
from insights_modules import budget_pipeline as pipeline   # type: ignore
from insights_modules import database as db                # type: ignore
from insights_modules.analysis_engine import CATEGORY_ARABIC  # type: ignore
from insights_modules.smart_insights_core import compute_features, detect_signals, generate_smart_insights, get_category_breakdown  # type: ignore
from insights_modules.achievements_and_reports import generate_achievements, generate_monthly_report # type: ignore


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #1e2736; }
h1, h2, h3 { font-family: 'Segoe UI', 'Cairo', sans-serif; }

.metric-tile {
    background: #161b27; border: 1px solid #1e2736;
    border-radius: 12px; padding: 18px 20px; margin-bottom: 10px;
}
.metric-tile .label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing:.08em; }
.metric-tile .value { font-size: 26px; font-weight: 700; color: #f9fafb; margin-top: 4px; }
.metric-tile .delta { font-size: 13px; margin-top: 4px; }

.badge-normal    { color:#10b981; background:#052e1c; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }
.badge-over      { color:#f59e0b; background:#2d1f02; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }
.badge-high-over { color:#ef4444; background:#2d0a0a; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; }

.rec-box { background:#1a2035; border-left:4px solid #3b82f6; border-radius:0 8px 8px 0;
           padding:14px 18px; font-size:14px; color:#d1d5db; line-height:1.6; margin-top:8px; white-space:pre-wrap; }
.rec-box.over      { border-left-color:#f59e0b; }
.rec-box.high_over { border-left-color:#ef4444; }
.rec-box.normal    { border-left-color:#10b981; }

.section-header { font-size:18px; font-weight:700; color:#e5e7eb;
    border-bottom:1px solid #1e2736; padding-bottom:8px; margin:24px 0 16px; }

.banner-success { background:#052e1c; border:1px solid #10b981; border-radius:8px; padding:12px 18px; color:#10b981; }
.banner-warning { background:#2d1f02; border:1px solid #f59e0b; border-radius:8px; padding:12px 18px; color:#f59e0b; }
.banner-error   { background:#2d0a0a; border:1px solid #ef4444; border-radius:8px; padding:12px 18px; color:#ef4444; }
.banner-info    { background:#0d1f35; border:1px solid #3b82f6; border-radius:8px; padding:12px 18px; color:#93c5fd; }
</style>
"""

STATUS_COLORS = {"normal": "#10b981", "over": "#f59e0b", "high_over": "#ef4444"}
STATUS_LABELS = {"normal": "✅ على المسار", "over": "⚠️ تجاوز المتوقع", "high_over": "🔴 تجاوز حرج"}

DEFAULT_PLAN = {"food": 3000.0, "saving": 2000.0, "emergency": 1000.0, "enjoyment": 1500.0}

ALL_CATEGORIES = [
    "Food", "Transport", "Rent", "Electricity", "Water", "Gas",
    "Internet", "Mobile", "Shopping", "Education", "Medical",
    "Entertainment", "Coffee", "Other", "Emergency",
]

PARENT_CAT_LABELS = {
    "enjoyment": "🎯 الاستمتاع",
    "food":      "🍽️ الغذاء",
    "saving":    "💰 الادخار",
    "emergency": "🏥 الطوارئ",
}


def _display_cat(cat: str) -> str:
    return CATEGORY_ARABIC.get(cat.lower(), cat.capitalize())


def render():
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 💰 متابع الميزانية الذكي")
        st.markdown("---")
        user_id = st.text_input("👤 معرف المستخدم", value="user_001", help="أدخل أي معرف فريد")

        st.markdown("---")
        st.markdown("### 📅 الشهر واليوم")
        now = datetime.today()
        months_list = [(i, n) for i, n in enumerate(
            ["يناير","فبراير","مارس","أبريل","مايو","يونيو",
             "يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"], 1)]
        month_names = {m[0]: m[1] for m in months_list}

        if "sel_month" not in st.session_state: st.session_state.sel_month = now.month
        if "sel_year"  not in st.session_state: st.session_state.sel_year  = now.year
        if "sel_day"   not in st.session_state: st.session_state.sel_day   = min(now.day, 30)

        sel_month = st.selectbox("الشهر", [m[0] for m in months_list],
                                  index=st.session_state.sel_month - 1,
                                  format_func=lambda x: f"{month_names[x]} {st.session_state.sel_year}",
                                  key="month_sel_ins")
        if sel_month != st.session_state.sel_month:
            st.session_state.sel_month = sel_month

        day = st.slider("يوم الشهر", 1, 30, st.session_state.sel_day, key="day_slider_ins")
        st.session_state.sel_day = day
        st.caption(f"📅 {month_names[sel_month]} — اليوم **{day}** من 30")

        st.markdown("---")
        st.markdown("### 📋 خطة الميزانية الشهرية (ج.م)")
        plan: dict[str, float] = {}
        for cat, default_val in DEFAULT_PLAN.items():
            label_ar = CATEGORY_ARABIC.get(cat.lower(), cat.capitalize())
            plan[cat] = st.number_input(label_ar, min_value=0.0, max_value=100_000.0,
                                         value=default_val, step=100.0, format="%.0f")

        st.markdown("---")
        st.markdown("### 🏠 المصروفات الثابتة")
        fixed_expenses: dict[str, float] = {}
        fixed_cats   = ["rent","electricity","water","gas","internet","mobile","transport","medical","education"]
        fixed_labels = {"rent":"الإيجار","electricity":"الكهرباء","water":"المياه","gas":"الغاز",
                        "internet":"الإنترنت","mobile":"التليفون","transport":"المواصلات",
                        "medical":"طبي","education":"تعليم"}
        fc1, fc2 = st.columns(2)
        for idx, cat in enumerate(fixed_cats):
            col = fc1 if idx % 2 == 0 else fc2
            with col:
                val = st.number_input(fixed_labels.get(cat, cat), min_value=0.0, max_value=50_000.0,
                                       value=0.0, step=50.0, format="%.0f", key=f"ins_fixed_{cat}")
                if val > 0:
                    fixed_expenses[cat] = val

        st.markdown("---")
        income = st.number_input("💵 الدخل الشهري (ج.م)", min_value=0.0, max_value=500_000.0,
                                  value=10_000.0, step=500.0, format="%.0f")

        if st.button("🗑️ حذف جميع مصروفاتي", use_container_width=True, type="secondary"):
            deleted = pipeline.reset_user(user_id)
            st.success(f"✅ تم حذف {deleted} سجل.")

    # ── Main header ───────────────────────────────────────────────────────────
    st.markdown("# 📊 لوحة متابعة الميزانية الذكية")
    st.markdown(
        f"**المستخدم:** `{user_id}` &nbsp;|&nbsp; "
        f"**اليوم:** {day}/30 &nbsp;|&nbsp; "
        f"**إجمالي الميزانية:** {sum(plan.values()):,.0f} ج.م"
    )

    # ── Add expense form ──────────────────────────────────────────────────────
    with st.expander("➕ تسجيل مصروف جديد", expanded=True):
        ac1, ac2, ac3, ac4 = st.columns([2, 2, 2, 1])
        with ac1:
            exp_category = st.selectbox("الفئة", ALL_CATEGORIES,
                                         format_func=lambda x: CATEGORY_ARABIC.get(x.lower(), x))
        with ac2:
            exp_amount = st.number_input("المبلغ (ج.م)", min_value=1.0, max_value=50_000.0, value=100.0, step=10.0)
        with ac3:
            exp_date = st.date_input("التاريخ", value=datetime.today())
        with ac4:
            st.write(""); st.write("")
            add_btn = st.button("💾 إضافة", use_container_width=True, type="primary")

        if add_btn:
            try:
                row_id = pipeline.record_expense(user_id=user_id, category=exp_category,
                                                  amount=exp_amount, date=exp_date.strftime("%Y-%m-%d"))
                cat_ar = CATEGORY_ARABIC.get(exp_category.lower(), exp_category)
                st.session_state["_last_expense"] = (
                    f'<div class="banner-success">✅ تم تسجيل المصروف — '
                    f'{exp_amount:,.0f} ج.م في <b>{cat_ar}</b> (سجل #{row_id})</div>'
                )
                st.rerun()
            except ValueError as e:
                st.markdown(f'<div class="banner-error">❌ خطأ: {e}</div>', unsafe_allow_html=True)

        if "_last_expense" in st.session_state:
            st.markdown(st.session_state.pop("_last_expense"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Pipeline data ─────────────────────────────────────────────────────────
    current_year = st.session_state.sel_year
    spending_data = db.get_monthly_spending(user_id, sel_month, current_year)
    status_data   = pipeline.check_user_status(user_id=user_id, plan=plan, day=day,
                                                month=sel_month, year=current_year)
    features  = compute_features(plan, fixed_expenses, spending_data, income, day)
    signals   = detect_signals(features)

    # ── KPI strip ────────────────────────────────────────────────────────────
    total_planned  = sum(plan.values())
    total_spent    = sum(d["spent"]    for d in status_data.values())
    total_expected = sum(d["expected"] for d in status_data.values())
    total_remaining= total_planned - total_spent
    health_pct     = (total_spent / total_expected * 100) if total_expected > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    def kpi_tile(col, label, value, delta=None, positive=True):
        with col:
            color      = "#10b981" if positive else "#ef4444"
            delta_html = f'<div class="delta" style="color:{color}">{delta}</div>' if delta else ""
            st.markdown(f"""
            <div class="metric-tile">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                {delta_html}
            </div>""", unsafe_allow_html=True)

    kpi_tile(kpi1, "إجمالي المُنفق",    f"{total_spent:,.0f} ج.م")
    kpi_tile(kpi2, "المتوقع حتى الآن", f"{total_expected:,.0f} ج.م", f"مسار اليوم {day}/30", True)
    kpi_tile(kpi3, "المتبقي",           f"{total_remaining:,.0f} ج.م",
             "⚠️ تجاوز" if total_remaining < 0 else "ضمن الميزانية", total_remaining >= 0)
    kpi_tile(kpi4, "صحة الميزانية",    f"{health_pct:.0f}٪",
             "طبيعي" if health_pct <= 100 else "تجاوز الوتيرة", health_pct <= 100)

    # ── Anomaly banners ───────────────────────────────────────────────────────
    for _cat, _result in pipeline.get_anomalies(user_id, plan).items():
        if not _result.message: continue
        css = "banner-warning" if _result.is_anomaly else "banner-info"
        st.markdown(f'<div class="{css}">{_result.message}</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Category cards ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📂 تفاصيل كل فئة</div>', unsafe_allow_html=True)
    breakdown = get_category_breakdown(features)

    for category, data in status_data.items():
        spent, expected, planned = data["spent"], data["expected"], data["planned"]
        remaining, status, rec   = data["remaining"], data["status"], data["recommendation"]
        status_color = STATUS_COLORS.get(status, "#6b7280")
        status_label = STATUS_LABELS.get(status, status)
        cat_ar       = PARENT_CAT_LABELS.get(category.lower(), _display_cat(category))
        bar_pct      = min(spent / planned * 100, 100) if planned > 0 else 0

        with st.container():
            h_left, h_right = st.columns([3, 1])
            with h_left: st.markdown(f"### {cat_ar}")
            with h_right:
                st.markdown(f'<span class="badge-{status.replace("_","-")}">{status_label}</span>',
                            unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("المُنفق",  f"{spent:,.0f} ج.م")
            m2.metric("المتوقع", f"{expected:,.0f} ج.م",
                      delta=f"{data['pct_of_expected']:.0f}٪ من الهدف")
            m3.metric("المخطط",  f"{planned:,.0f} ج.م")
            m4.metric("المتبقي", f"{remaining:,.0f} ج.م",
                      delta="⚠️ تجاوز" if remaining < 0 else None)

            expected_pct = min(expected / planned * 100, 100) if planned > 0 else 0
            fig = go.Figure(go.Bar(x=[bar_pct], orientation="h",
                                    marker_color=status_color, width=0.5,
                                    text=[f"{bar_pct:.1f}٪"], textposition="outside"))
            fig.add_vline(x=expected_pct, line_dash="dash", line_color="#6b7280",
                          annotation_text=f"المتوقع: {expected_pct:.0f}٪",
                          annotation_position="top right",
                          annotation_font_color="#9ca3af")
            fig.update_layout(
                height=80, margin=dict(l=0, r=80, t=10, b=0),
                xaxis=dict(range=[0, 110], showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key=f"ins_bar_{category}")
            st.markdown(f'<div class="rec-box {status}">{rec}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Smart insights ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧠 الرؤى الذكية</div>', unsafe_allow_html=True)
    insights = generate_smart_insights(signals, features, day)
    if not insights:
        st.info("💡 أضف بعض المصروفات لرؤية رؤى ذكية حول ميزانيتك.")
    else:
        for ins in insights[:8]:
            css = {"high": "banner-error", "medium": "banner-warning"}.get(
                ins.get("priority", "medium"), "banner-info"
            )
            st.markdown(f'<div class="{css}">{ins["message"]}</div>', unsafe_allow_html=True)
            st.markdown("")

    # ── Financial Score & Report ──────────────────────────────────────────────
    
    st.markdown('<div class="section-header">📜 التقرير الشهري ودرجة الصحة المالية</div>', unsafe_allow_html=True)
    report = generate_monthly_report(features, signals)
    st.markdown(f"**ملخص الأداء:** {report.narrative}")
    
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**🌟 نقاط القوة**")
        for s in report.strengths:
            st.markdown(f"- ✅ {s}")
    with r2:
        st.markdown("**⚠️ تحذيرات ورؤى**")
        for w in report.warnings:
            st.markdown(f"- 🔴 {w}")
        for w in getattr(report, "weaknesses", []): # Use weaknesses if present, fallback handled
            st.markdown(f"- ⚠️ {w}")
        for w in report.insights:
            st.markdown(f"- 💡 {w}")
            
    st.markdown("---")

    # ── Achievements ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 الإنجازات</div>', unsafe_allow_html=True)
    history_data = db.get_monthly_summaries(user_id, current_year)
    ach_result = generate_achievements(features, signals, fixed_expenses, history_data)
    
    if ach_result.new_achievements:
        new_titles = ", ".join([a['title'] for a in ach_result.new_achievements])
        st.success(f"🎉 مبروك! لقد حققت إنجازات جديدة: {new_titles}")
        
    ach_cols = st.columns(4)
    for idx, ach in enumerate(ach_result.all_achievements):
        col = ach_cols[idx % 4]
        with col:
            opacity = "1.0" if ach["earned"] else "0.3"
            status_text = "مكتسب" if ach["earned"] else "مقفول"
            st.markdown(f'''
            <div style="opacity: {opacity}; text-align: center; padding: 10px; background: #161b27; border: 1px solid #1e2736; border-radius: 8px; margin-bottom: 10px;">
                <div style="font-size: 30px;">{ach['icon']}</div>
                <div style="font-weight: bold; margin-top: 5px;">{ach['title']}</div>
                <div style="font-size: 10px; color: #9ca3af;">{status_text}</div>
            </div>
            ''', unsafe_allow_html=True)
            
    st.markdown("---")

    # ── Weekly summary ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 ملخص الأسبوع (آخر 7 أيام)</div>', unsafe_allow_html=True)
    st.code(pipeline.get_weekly_summary(user_id, plan), language=None)

    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#475569; font-size:0.8rem">💰 نظام متابعة الميزانية الذكي</div>',
                unsafe_allow_html=True)
