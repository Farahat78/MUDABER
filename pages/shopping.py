"""
pages/shopping.py
────────────────────────────────────────────────────────────────────────────────
Smart Shopping List page.

IMPORTANT RULES (enforced here):
  ✅ Reads product data from data/latest_products.csv via core.data_loader.
  ✅ Reads price predictions from data/predictions.csv via core.data_loader.
  ✅ Uses shopping_modules/ for all list generation logic.
  ❌ Does NOT scrape any data.
  ❌ Does NOT run any ML training.
  ❌ Does NOT generate the CSV files (they come from the pipeline).
"""

from __future__ import annotations

import os
import sys
import uuid
import random
from io import StringIO

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.data_loader import load_products, load_predictions, get_data_freshness
from core.constants import TREND_COLORS, TREND_LABELS_AR, CATEGORY_ARABIC

# ── Shopping module imports (from shopping_modules/) ──────────────────────────
_SM = os.path.join(os.path.dirname(__file__), "..", "shopping_modules")
sys.path.insert(0, _SM)

from preprocessing import load_and_clean          # type: ignore
from generator import generate_smart_shopping_list, get_list_summary  # type: ignore
from feedback import process_modification         # type: ignore
from user_memory import load_memory, get_generation_hints, clear_memory  # type: ignore
from utils import format_currency, BUDGET_ALLOCATIONS  # type: ignore


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
.shop-title { font-size:1.6rem; font-weight:800; color:#f1f5f9; }
.shop-sub   { color:#94a3b8; font-size:0.9rem; margin-bottom:1rem; }
.cat-header {
    background: #1e293b; border-left: 4px solid #3b82f6;
    padding: 0.4rem 0.8rem; border-radius: 4px;
    font-weight: 700; color: #e2e8f0; margin-top: 1.2rem;
}
.trend-up   { color: #ef4444; font-weight: 700; }
.trend-down { color: #10b981; font-weight: 700; }
.trend-stable { color: #94a3b8; }
.data-badge {
    display: inline-block; background: #1e293b;
    border: 1px solid #334155; border-radius: 20px;
    padding: 2px 10px; font-size: 0.75rem; color: #64748b;
}
</style>
"""


def _enrich_with_predictions(sl: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Join predictions onto the shopping list (best-effort fuzzy match on name)."""
    # Drop columns if they already exist to prevent duplicates when called multiple times
    for col in ["trend_label", "predicted_price", "change_percentage"]:
        if col in sl.columns:
            sl = sl.drop(columns=[col])

    if preds.empty or sl.empty:
        sl["trend_label"] = "STABLE"
        sl["predicted_price"] = None
        sl["change_percentage"] = 0.0
        return sl

    preds_map = preds.set_index("product_name_clean")[
        ["trend_label", "predicted_price", "change_percentage"]
    ].to_dict("index")

    def _lookup(name: str) -> tuple:
        key = str(name).lower().strip()
        if key in preds_map:
            r = preds_map[key]
            return r["trend_label"], r["predicted_price"], r["change_percentage"]
        return "STABLE", None, 0.0

    results = sl["product_name"].apply(lambda n: pd.Series(_lookup(n),
                  index=["trend_label", "predicted_price", "change_percentage"]))
    return pd.concat([sl, results], axis=1)


def render():
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="shop-title">🛒 قائمة التسوق الذكية</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="shop-sub">توليد قائمة تسوق شهرية مثالية من بيانات كارفور مصر — مع توقعات الأسعار من الـ Pipeline</div>',
        unsafe_allow_html=True,
    )

    # ── Session state ─────────────────────────────────────────────────────────
    if "shop_session_id" not in st.session_state:
        st.session_state.shop_session_id = str(uuid.uuid4())[:8]
    if "shopping_list" not in st.session_state:
        st.session_state.shopping_list = None
    if "mod_history" not in st.session_state:
        st.session_state.mod_history = []

    # ── Load data (from CSV, not from scraper) ────────────────────────────────
    with st.spinner("⏳ جارٍ تحميل بيانات المنتجات..."):
        raw_products = load_products()
        predictions  = load_predictions()

    if raw_products.empty:
        st.error("❌ لم يتم العثور على بيانات المنتجات. تأكد من تشغيل الـ Pipeline أولاً.")
        st.info("📁 المتوقع: `data/latest_products.csv`")
        return

    # Clean the products using the shopping module preprocessor
    @st.cache_data(show_spinner=False)
    def _clean(path_hash: int) -> pd.DataFrame:
        return load_and_clean(raw_products)   # pass DataFrame directly if supported

    df_clean = load_and_clean(raw_products)

    # ── Freshness badge ───────────────────────────────────────────────────────
    freshness = get_data_freshness()
    st.markdown(
        f'<span class="data-badge">📦 {len(df_clean):,} منتج</span> &nbsp;'
        f'<span class="data-badge">🕐 آخر تحديث: {freshness.get("products","—")}</span> &nbsp;'
        f'<span class="data-badge">📈 {len(predictions):,} توقع سعري</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ إعدادات التسوق")
        monthly_budget = st.number_input("💰 الميزانية الشهرية (ج.م)", min_value=500, max_value=50000, value=3000, step=100)
        family_size    = st.slider("👨‍👩‍👧‍👦 عدد أفراد الأسرة", 1, 10, 4)
        lifestyle      = st.selectbox("🏠 مستوى المعيشة", ["low", "medium", "high"],
                                       format_func=lambda x: {"low": "🟢 اقتصادي", "medium": "🟡 متوسط", "high": "🔴 مرتفع"}[x],
                                       index=1)
        randomize      = st.checkbox("🎲 توليد عشوائي", value=True)

        st.divider()
        st.markdown("### 📊 توزيع الميزانية المقترح")
        alloc = BUDGET_ALLOCATIONS[lifestyle]
        alloc_data = [{"فئة": cat, "نسبة": f"{pct*100:.0f}%", "مبلغ": f"{monthly_budget*pct:.0f} ج.م"}
                      for cat, pct in alloc.items() if pct > 0]
        st.dataframe(pd.DataFrame(alloc_data), hide_index=True, use_container_width=True)

        st.divider()
        memory = load_memory(st.session_state.shop_session_id)
        if memory.has_preferences:
            st.markdown("### 🧠 ذاكرة التفضيلات")
            if memory.avoided_items:
                st.markdown(f"🚫 **متجنب:** {', '.join(memory.avoided_items[:4])}")
            if memory.preferred_items:
                st.markdown(f"✅ **مفضل:** {', '.join(memory.preferred_items[:4])}")
            if st.button("🗑️ مسح الذاكرة", use_container_width=True):
                clear_memory(st.session_state.shop_session_id)
                st.rerun()

    # ── Generate / Regenerate ─────────────────────────────────────────────────
    gen_col, regen_col = st.columns([3, 1])
    with gen_col:
        gen_clicked   = st.button("🚀 توليد قائمة التسوق الذكية", type="primary", use_container_width=True)
    with regen_col:
        regen_clicked = st.button("🔄 إعادة التوليد", use_container_width=True,
                                   disabled=st.session_state.shopping_list is None)

    if gen_clicked or regen_clicked:
        with st.spinner("🧠 النظام يُحلل البيانات ويولّد القائمة المثلى..."):
            seed = None if randomize else (42 if not regen_clicked else random.randint(0, 9999))
            hints = get_generation_hints(st.session_state.shop_session_id)
            sl = generate_smart_shopping_list(
                df=df_clean,
                monthly_budget=monthly_budget,
                family_size=family_size,
                lifestyle=lifestyle,
                random_seed=seed,
                memory_skip_norms=hints.skip_norms,
            )
            sl = _enrich_with_predictions(sl, predictions)
            st.session_state.shopping_list = sl
            st.session_state.mod_history   = []
        st.success("✅ تمّ توليد القائمة بنجاح!")

    # ── Display list ──────────────────────────────────────────────────────────
    if st.session_state.shopping_list is not None:
        sl      = st.session_state.shopping_list
        summary = get_list_summary(sl, monthly_budget)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📦 إجمالي المنتجات", summary.get("total_items", 0))
        m2.metric("💵 إجمالي التكلفة",  format_currency(summary.get("total_spent", 0)))
        pct = summary.get("budget_used_pct", 0)
        m3.metric("📊 نسبة الميزانية",  f"{pct}%")
        m4.metric("💰 المتبقي",          format_currency(summary.get("remaining", 0)))

        st.divider()

        # Charts
        ch1, ch2 = st.columns(2)
        with ch1:
            by_cat = sl.groupby("category")["total_price"].sum().reset_index()
            fig_pie = px.pie(by_cat, values="total_price", names="category",
                             title="توزيع الإنفاق حسب الفئة", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font=dict(color="#e2e8f0"), height=300, margin=dict(t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with ch2:
            trend_counts = sl["trend_label"].value_counts().reset_index()
            trend_counts.columns = ["trend", "count"]
            trend_counts["label"] = trend_counts["trend"].map(TREND_LABELS_AR)
            trend_counts["color"] = trend_counts["trend"].map(TREND_COLORS)
            fig_trend = px.bar(trend_counts, x="label", y="count",
                               title="توزيع توقعات الأسعار",
                               color="trend",
                               color_discrete_map=TREND_COLORS)
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#e2e8f0"), height=300,
                                    showlegend=False, margin=dict(t=40, b=0))
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

        st.divider()

        # Detailed table by category
        st.markdown("## 🛒 قائمة التسوق المفصّلة")
        display_cols   = ["product_name", "unit_price", "quantity", "total_price", "trend_label", "slot"]
        display_labels = {
            "product_name": "المنتج", "unit_price": "سعر الوحدة",
            "quantity": "الكمية", "total_price": "الإجمالي",
            "trend_label": "توقع السعر", "slot": "النوع",
        }

        for cat in sl["category"].unique():
            cat_df = sl[sl["category"] == cat].copy()
            cat_label = CATEGORY_ARABIC.get(cat.lower(), cat)
            st.markdown(
                f'<div class="cat-header">{cat_label} '
                f'<span style="font-weight:400; font-size:0.85rem; color:#94a3b8;">'
                f'({len(cat_df)} منتجات — {format_currency(cat_df["total_price"].sum())})'
                f'</span></div>',
                unsafe_allow_html=True,
            )
            avail_cols = [c for c in display_cols if c in cat_df.columns]
            display = cat_df[avail_cols].rename(columns=display_labels).copy()
            if "سعر الوحدة" in display.columns:
                display["سعر الوحدة"] = display["سعر الوحدة"].apply(lambda x: f"{x:.2f} ج.م")
            if "الإجمالي" in display.columns:
                display["الإجمالي"] = display["الإجمالي"].apply(lambda x: f"{x:.2f} ج.م")
            if "توقع السعر" in display.columns:
                display["توقع السعر"] = display["توقع السعر"].map(TREND_LABELS_AR).fillna("➡️ مستقر")
            if "النوع" in display.columns:
                display["النوع"] = display["النوع"].apply(
                    lambda s: "✅ أساسي" if s != "filler" else "➕ إضافي"
                )
            st.dataframe(display, hide_index=True, use_container_width=True)

        st.divider()

        # NLP modification
        st.markdown("## 🤖 تعديل بالغة الطبيعية")
        nlp_col, apply_col = st.columns([5, 1])
        with nlp_col:
            user_instruction = st.text_input("اكتب طلبك:", placeholder="مثال: مش عايز غير لبن وجبنة",
                                              label_visibility="collapsed", key="nlp_shop")
        with apply_col:
            apply_mod = st.button("✨ تطبيق", type="primary", use_container_width=True)

        if apply_mod and user_instruction.strip():
            with st.spinner("⚙️ جاري التحليل..."):
                updated, mod_summary, score_bd, parsed_intent = process_modification(
                    instruction=user_instruction.strip(), shopping_list=sl,
                    dataset=df_clean, budget=monthly_budget,
                    session_id=st.session_state.shop_session_id,
                )
            if updated is not None and len(updated) > 3:
                st.session_state.shopping_list = _enrich_with_predictions(updated, predictions)
                st.success(mod_summary)
                st.rerun()
            else:
                st.warning(mod_summary)

        st.divider()

        # Export
        st.markdown("## 💾 تصدير القائمة")
        buf = StringIO()
        sl.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 تنزيل CSV",
            data=buf.getvalue().encode("utf-8-sig"),
            file_name=f"shopping_list_{st.session_state.shop_session_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )
