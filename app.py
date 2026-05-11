"""
app.py — Smart Salary System
==============================
Single Streamlit entry point.  Provides top-level navigation between the
three sub-apps (Financial, Shopping, Insights).

Run:
    streamlit run app.py
"""

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Smart Salary System | نظام الراتب الذكي",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', 'Inter', sans-serif;
}

/* Sidebar nav buttons */
.nav-btn {
    display: block;
    width: 100%;
    padding: 0.75rem 1rem;
    margin: 0.25rem 0;
    border-radius: 10px;
    border: none;
    background: transparent;
    text-align: right;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    color: #94a3b8;
    transition: all 0.2s ease;
}
.nav-btn:hover { background: #1e293b; color: #f1f5f9; }
.nav-btn.active { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; }

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Feature card */
.feature-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    height: 100%;
}
.feature-card:hover {
    transform: translateY(-4px);
    border-color: #3b82f6;
}
.feature-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.feature-title { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; }
.feature-desc { font-size: 0.875rem; color: #94a3b8; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Navigation state ──────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💡 نظام الراتب الذكي")
    st.markdown("---")

    nav_items = [
        ("🏠", "الرئيسية",         "home"),
        ("💰", "النموذج المالي",   "financial"),
        ("🛒", "قائمة التسوق",    "shopping"),
        ("📊", "متابع الميزانية",  "insights"),
    ]
    for icon, label, key in nav_items:
        active_cls = "active" if st.session_state.page == key else ""
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.markdown("---")
    st.caption("v1.0 — Data via GitHub")

# ── Page router ───────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "financial":
    from pages.financial import render
    render()

elif page == "shopping":
    from pages.shopping import render
    render()

elif page == "insights":
    from pages.insights import render
    render()

else:
    # ── Home / Landing ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">💡 Smart Salary System</div>
        <div class="hero-title" style="font-size:1.8rem;">نظام إدارة الراتب الذكي</div>
        <p class="hero-sub">
            منظومة متكاملة لتخطيط الراتب، توليد قائمة التسوق، ومتابعة الميزانية —
            مدعومة بالذكاء الاصطناعي وبيانات كارفور الحقيقية.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">النموذج المالي</div>
            <div class="feature-desc">
                أدخل بيانات راتبك والتزاماتك — يوزّع نموذج XGBoost دخلك
                المتاح على الغذاء، الادخار، الطوارئ، والاختياري.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 ابدأ التخطيط المالي", key="go_financial", use_container_width=True):
            st.session_state.page = "financial"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🛒</div>
            <div class="feature-title">قائمة التسوق الذكية</div>
            <div class="feature-desc">
                يولّد قائمة تسوق شهرية مثالية من بيانات كارفور مصر الحقيقية،
                مع تعديل بالغة الطبيعية وتوقعات الأسعار.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🛒 توليد قائمة التسوق", key="go_shopping", use_container_width=True):
            st.session_state.page = "shopping"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">متابع الميزانية</div>
            <div class="feature-desc">
                سجّل مصروفاتك اليومية وتابع ميزانيتك بالرؤى الذكية،
                الإنجازات، ومحرك التحسين الرياضي.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 فتح متابع الميزانية", key="go_insights", use_container_width=True):
            st.session_state.page = "insights"
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.85rem; padding:1rem">
        📡 البيانات تُحدَّث تلقائياً عبر Pipeline → GitHub → Streamlit Cloud<br>
        لا يوجد اتصال مباشر بين الواجهة والسكريبتات — كل شيء يعمل عبر ملفات CSV فقط.
    </div>
    """, unsafe_allow_html=True)
