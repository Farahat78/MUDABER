"""
pages/financial.py
────────────────────────────────────────────────────────────────────────────────
Financial Model page — Smart Salary Allocation using XGBoost.

IMPORTANT RULES (enforced here):
  ✅ Reads user inputs from Streamlit widgets.
  ✅ Calls core/financial_engine.py for ML inference.
  ❌ Does NOT scrape any data.
  ❌ Does NOT run model training.
  ❌ Does NOT write any CSV files.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Ensure core is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.financial_engine import calculate_features, predict_ratios, build_allocation


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
.fin-header { font-size:1.6rem; font-weight:800; color:#f1f5f9; margin-bottom:0.25rem; }
.fin-sub    { color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem; }
.alloc-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.alloc-icon  { font-size: 2rem; }
.alloc-label { font-size: 0.85rem; color: #94a3b8; margin: 0.4rem 0 0.2rem; }
.alloc-value { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; }
.alloc-ratio { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 1px solid #334155; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem;
}
</style>
"""


def render():
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="fin-header">💰 النموذج المالي الذكي</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fin-sub">أدخل بياناتك المالية ويوزّع نموذج XGBoost دخلك المتاح بذكاء</div>',
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("financial_form"):
        st.markdown('<div class="section-title">👤 المعلومات الأساسية</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_salary = st.number_input("💵 الراتب الشهري (ج.م)", min_value=0.0, value=10000.0, step=500.0)
        with c2:
            number_of_family_members = st.number_input("👨‍👩‍👧‍👦 عدد أفراد الأسرة", min_value=1, max_value=15, value=3)
        with c3:
            marital_status = st.selectbox("💍 الحالة الاجتماعية", ["single", "married"],
                                          format_func=lambda x: "أعزب" if x == "single" else "متزوج")

        c4, c5, c6 = st.columns(3)
        with c4:
            living_cost_level = st.selectbox("🏘️ مستوى تكلفة المعيشة", ["Low", "Medium", "High"],
                                              format_func=lambda x: {"Low": "منخفض", "Medium": "متوسط", "High": "مرتفع"}[x],
                                              index=1)
        with c5:
            income_stability = st.slider("📈 استقرار الدخل (0=متقلب، 1=ثابت)", 0.0, 1.0, 0.7)
        with c6:
            saving_preference = st.selectbox("💰 تفضيل الادخار", ["Low", "Medium", "High"],
                                              format_func=lambda x: {"Low": "منخفض", "Medium": "متوسط", "High": "مرتفع"}[x],
                                              index=1)

        st.markdown('<div class="section-title">🎯 التفضيلات والأولويات</div>', unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        with c7:
            risk_tolerance = st.selectbox("⚖️ تحمّل المخاطر", ["Low", "Medium", "High"],
                                           format_func=lambda x: {"Low": "منخفض", "Medium": "متوسط", "High": "مرتفع"}[x],
                                           index=1)
        PRIORITY_OPTIONS = ["Food", "Emergency", "Optional", "Savings"]
        PRIORITY_LABELS  = {"Food": "🍽️ غذاء", "Emergency": "🚨 طوارئ",
                            "Optional": "🎮 اختياري", "Savings": "💰 ادخار"}
        pc1, pc2, pc3, pc4 = st.columns(4)
        priority_1 = pc1.selectbox("الأولوية 1", PRIORITY_OPTIONS, format_func=lambda x: PRIORITY_LABELS[x], index=0)
        priority_2 = pc2.selectbox("الأولوية 2", PRIORITY_OPTIONS, format_func=lambda x: PRIORITY_LABELS[x], index=3)
        priority_3 = pc3.selectbox("الأولوية 3", PRIORITY_OPTIONS, format_func=lambda x: PRIORITY_LABELS[x], index=1)
        priority_4 = pc4.selectbox("الأولوية 4", PRIORITY_OPTIONS, format_func=lambda x: PRIORITY_LABELS[x], index=2)

        st.markdown('<div class="section-title">🏠 المصروفات الثابتة</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        rent                 = e1.number_input("🏠 الإيجار", value=0.0, step=100.0)
        utilities            = e2.number_input("💡 المرافق", value=0.0, step=50.0)
        transportation       = e3.number_input("🚗 المواصلات", value=0.0, step=50.0)
        optional_services_cost = e4.number_input("📱 خدمات اختيارية", value=0.0, step=50.0)

        st.markdown('<div class="section-title">📋 الديون والنفقات السنوية (JSON)</div>', unsafe_allow_html=True)
        dj1, dj2 = st.columns(2)
        monthly_debts_json  = dj1.text_area("الديون الشهرية", value='[]',
                                             help='مثال: [{"name":"قرض","amount":500}]', height=80)
        annual_expenses_json = dj2.text_area("النفقات السنوية", value='[]',
                                              help='مثال: [{"name":"تأمين","amount":3000}]', height=80)

        submitted = st.form_submit_button("🔮 احسب التوزيع المثالي", type="primary", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if submitted:
        try:
            monthly_debts  = json.loads(monthly_debts_json)
            annual_expenses = json.loads(annual_expenses_json)
        except json.JSONDecodeError as e:
            st.error(f"❌ خطأ في JSON: {e}")
            return

        user_data = pd.DataFrame([{
            "monthly_salary":         monthly_salary,
            "number_of_family_members": number_of_family_members,
            "marital_status":          marital_status,
            "living_cost_level":       living_cost_level,
            "income_stability":        income_stability,
            "saving_preference":       saving_preference,
            "risk_tolerance":          risk_tolerance,
            "priority_1":              priority_1,
            "priority_2":              priority_2,
            "priority_3":              priority_3,
            "priority_4":              priority_4,
            "rent":                    rent,
            "utilities":               utilities,
            "transportation":          transportation,
            "optional_services_cost":  optional_services_cost,
            "monthly_debts_json":      monthly_debts,
            "annual_expenses_json":    annual_expenses,
            "services_percentage":     0,
        }])

        with st.spinner("🧠 النموذج يحسب التوزيع المثالي..."):
            user_data, feature_cols = calculate_features(user_data)
            ratios = predict_ratios(user_data, feature_cols)
            available_income = float(user_data["available_income"].values[0])
            allocation = build_allocation(monthly_salary, ratios, available_income)

        if available_income <= 0:
            st.warning("⚠️ الدخل المتاح صفر أو سالب — تحقق من المصروفات الثابتة والديون.")
            return

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 نتيجة التوزيع الذكي")

        icons = {"food": "🍽️", "saving": "💰", "emergency": "🚨", "optional": "🎮"}
        labels = {"food": "الغذاء", "saving": "الادخار", "emergency": "الطوارئ", "optional": "الاختياري"}

        k0, k1, k2, k3, k4 = st.columns(5)
        k0.metric("💵 الدخل المتاح", f"{available_income:,.0f} ج.م")

        for col, cat in zip([k1, k2, k3, k4], ["food", "saving", "emergency", "optional"]):
            pct = allocation["ratios"][cat] * 100
            col.metric(
                f"{icons[cat]} {labels[cat]}",
                f"{allocation[cat]:,.0f} ج.م",
                f"{pct:.1f}%"
            )

        # ── Pie chart ─────────────────────────────────────────────────────────
        chart_col, table_col = st.columns([2, 1])
        with chart_col:
            pie_vals   = [allocation[c] for c in ["food", "saving", "emergency", "optional"]]
            pie_labels = [f"{icons[c]} {labels[c]}" for c in ["food", "saving", "emergency", "optional"]]
            pie_colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
            fig = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_vals,
                hole=0.5,
                marker_colors=pie_colors,
                textfont=dict(color="white", size=13),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                height=340,
                margin=dict(t=20, b=0),
                legend=dict(font=dict(color="#e2e8f0")),
                annotations=[dict(
                    text=f"متاح<br>{available_income:,.0f}",
                    x=0.5, y=0.5, font_size=14, font_color="#f1f5f9", showarrow=False
                )],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with table_col:
            rows = []
            for cat in ["food", "saving", "emergency", "optional"]:
                rows.append({
                    "الفئة":      f"{icons[cat]} {labels[cat]}",
                    "المبلغ (ج.م)": f"{allocation[cat]:,.0f}",
                    "النسبة %":   f"{allocation['ratios'][cat]*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # ── Bar chart: budget overview ─────────────────────────────────────────
        st.markdown("#### 📈 نظرة عامة على الميزانية")
        fixed_costs = rent + utilities + transportation + optional_services_cost
        debts       = sum(d.get("amount", 0) for d in monthly_debts)
        annual_mo   = sum(a.get("amount", 0) for a in annual_expenses) / 12

        bar_categories = ["الثابتة", "الديون", "سنوية/12", "غذاء", "ادخار", "طوارئ", "اختياري"]
        bar_values     = [fixed_costs, debts, annual_mo, allocation["food"],
                          allocation["saving"], allocation["emergency"], allocation["optional"]]
        bar_colors     = ["#64748b","#94a3b8","#cbd5e1","#3b82f6","#10b981","#f59e0b","#8b5cf6"]

        fig2 = px.bar(
            x=bar_categories, y=bar_values,
            color=bar_categories,
            color_discrete_sequence=bar_colors,
            labels={"x": "الفئة", "y": "المبلغ (ج.م)"},
        )
        fig2.add_hline(y=monthly_salary, line_dash="dash", line_color="#ef4444",
                       annotation_text=f"الراتب: {monthly_salary:,.0f}", annotation_font_color="#ef4444")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=300, margin=dict(t=20, b=0),
            xaxis=dict(color="#e2e8f0"), yaxis=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
