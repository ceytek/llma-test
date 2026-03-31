"""
POC: Llama 3 ile CV–İlan Eşleştirme
Tamamen local çalışır, Ollama üzerinden Llama 3 modeli kullanır.
"""
import json
import time
import streamlit as st

from llama_client import LlamaClient
from cv_extractor import extract_text
from prompts import (
    CV_PARSE_SYSTEM,
    MATCH_SYSTEM,
    build_cv_parse_prompt,
    build_match_prompt,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="POC – Llama 3 CV Analiz", page_icon="🦙", layout="wide")
st.title("🦙 Llama 3 – CV & İlan Eşleştirme POC")
st.caption("Tamamen local çalışır. Veri dışarı çıkmaz.")

# ── Sidebar: Ollama connection ────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Ollama Ayarları")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    client = LlamaClient(base_url=ollama_url)

    if client.is_available():
        models = client.list_models()
        if models:
            selected_model = st.selectbox("Model", models, index=0)
            client.model = selected_model
        else:
            st.warning("Ollama çalışıyor ama model bulunamadı. `ollama pull llama3` çalıştırın.")
            st.stop()
        st.success(f"Bağlı: {selected_model}")
    else:
        st.error("Ollama'ya bağlanılamıyor! `ollama serve` çalıştığından emin olun.")
        st.stop()

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    client.temperature = temperature

# ── Two-column layout ────────────────────────────────────────────────────────

col_job, col_cv = st.columns(2)

# ── LEFT: Job posting form ────────────────────────────────────────────────────

with col_job:
    st.header("📋 İlan Bilgileri")
    job_title = st.text_input("Pozisyon", placeholder="Örn: Senior Backend Developer")
    job_department = st.text_input("Departman", placeholder="Örn: Yazılım Geliştirme")
    job_location = st.text_input("Lokasyon", placeholder="Örn: İstanbul")
    job_experience = st.text_input("Deneyim Seviyesi", placeholder="Örn: 3-5 yıl")
    job_education = st.text_input("Eğitim", placeholder="Örn: Lisans - Bilgisayar Mühendisliği")
    job_description = st.text_area(
        "İlan Açıklaması",
        height=150,
        placeholder="Pozisyon gereksinimleri, sorumluluklar...",
    )
    job_keywords = st.text_input(
        "Aranan Yetenekler (virgülle)",
        placeholder="Python, FastAPI, PostgreSQL, Docker",
    )

# ── RIGHT: CV upload ─────────────────────────────────────────────────────────

with col_cv:
    st.header("📄 CV Yükle")
    uploaded_file = st.file_uploader(
        "PDF veya DOCX formatında CV seçin",
        type=["pdf", "docx", "doc"],
    )

    cv_text = ""
    if uploaded_file:
        with st.spinner("CV okunuyor..."):
            cv_text = extract_text(uploaded_file.read(), uploaded_file.name)
        st.success(f"CV okundu — {len(cv_text)} karakter")
        with st.expander("CV Metni (önizleme)"):
            st.text(cv_text[:2000] + ("..." if len(cv_text) > 2000 else ""))

# ── Analyze button ────────────────────────────────────────────────────────────

st.divider()

can_analyze = bool(job_title and job_description and cv_text)

if not can_analyze:
    st.info("Analiz başlatmak için ilan bilgilerini doldurun ve CV yükleyin.")

if st.button("🔍 Analiz Et", disabled=not can_analyze, type="primary", use_container_width=True):

    job_data = {
        "title": job_title,
        "department": job_department,
        "location": job_location,
        "experience_level": job_experience,
        "required_education": job_education,
        "description": job_description,
        "keywords": [k.strip() for k in job_keywords.split(",") if k.strip()],
    }

    # ── Step 1: Parse CV ──────────────────────────────────────────────────
    with st.status("🧠 CV analiz ediliyor...", expanded=True) as status:
        st.write("Llama 3 ile CV ayrıştırılıyor...")
        t0 = time.time()
        try:
            parsed_cv = client.chat_json(
                system_prompt=CV_PARSE_SYSTEM,
                user_prompt=build_cv_parse_prompt(cv_text),
            )
            parse_time = time.time() - t0
            st.write(f"CV ayrıştırma tamamlandı ({parse_time:.1f}s)")
        except Exception as e:
            st.error(f"CV ayrıştırma hatası: {e}")
            st.stop()

        # ── Step 2: Match CV to Job ───────────────────────────────────────
        st.write("CV–İlan eşleştirmesi yapılıyor...")
        t1 = time.time()
        try:
            match_prompt = build_match_prompt(job_data, parsed_cv)
            match_result = client.chat_json(
                system_prompt=MATCH_SYSTEM,
                user_prompt=match_prompt,
            )
            match_time = time.time() - t1
            st.write(f"Eşleştirme tamamlandı ({match_time:.1f}s)")
        except Exception as e:
            st.error(f"Eşleştirme hatası: {e}")
            st.stop()

        status.update(label="Analiz tamamlandı!", state="complete")

    # ── Results ───────────────────────────────────────────────────────────
    st.divider()
    st.header("📊 Sonuçlar")

    # Overall score + recommendation
    score = match_result.get("overall_score", 0)
    rec = match_result.get("recommendation", "N/A")

    rec_colors = {
        "highly_recommended": "🟢",
        "recommended": "🔵",
        "maybe": "🟡",
        "not_recommended": "🔴",
    }
    rec_labels = {
        "highly_recommended": "Kesinlikle Önerilir",
        "recommended": "Önerilir",
        "maybe": "Değerlendirilebilir",
        "not_recommended": "Önerilmez",
    }

    m1, m2, m3 = st.columns(3)
    m1.metric("Genel Puan", f"{score}/100")
    m2.metric("Öneri", f"{rec_colors.get(rec, '⚪')} {rec_labels.get(rec, rec)}")
    m3.metric("Toplam Süre", f"{parse_time + match_time:.1f}s")

    # Breakdown
    st.subheader("Puan Dağılımı")
    bd = match_result.get("breakdown", {})
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
    bc1.metric("Deneyim", f"{bd.get('experience_score', 0)}/30")
    bc2.metric("Eğitim", f"{bd.get('education_score', 0)}/20")
    bc3.metric("Yetenekler", f"{bd.get('skills_score', 0)}/30")
    bc4.metric("Dil", f"{bd.get('language_score', 0)}/10")
    bc5.metric("Uyum", f"{bd.get('fit_score', 0)}/10")

    # Reasoning
    with st.expander("Detaylı Değerlendirme", expanded=True):
        for key in ("experience", "education", "skills", "language", "fit"):
            reasoning = bd.get(f"{key}_reasoning", "")
            if reasoning:
                labels = {
                    "experience": "Deneyim",
                    "education": "Eğitim",
                    "skills": "Yetenekler",
                    "language": "Dil",
                    "fit": "Uyum",
                }
                st.markdown(f"**{labels[key]}:** {reasoning}")

    # Summary
    summary = match_result.get("summary", "")
    if summary:
        st.subheader("Özet")
        st.info(summary)

    # Skills
    sk1, sk2 = st.columns(2)
    with sk1:
        matched = match_result.get("matched_skills", [])
        if matched:
            st.subheader("✅ Eşleşen Yetenekler")
            st.write(", ".join(str(s) for s in matched))
    with sk2:
        missing = match_result.get("missing_skills", [])
        if missing:
            st.subheader("❌ Eksik Yetenekler")
            st.write(", ".join(str(s) for s in missing))

    # Strengths / Weaknesses
    sw1, sw2 = st.columns(2)
    with sw1:
        strengths = match_result.get("strengths", [])
        if strengths:
            st.subheader("💪 Güçlü Yanlar")
            for s in strengths:
                st.markdown(f"- {s}")
    with sw2:
        weaknesses = match_result.get("weaknesses", [])
        if weaknesses:
            st.subheader("⚠️ Zayıf Yanlar")
            for w in weaknesses:
                st.markdown(f"- {w}")

    # Raw JSON (collapsible)
    with st.expander("Ham CV Parse Sonucu (JSON)"):
        st.json(parsed_cv)
    with st.expander("Ham Eşleştirme Sonucu (JSON)"):
        st.json(match_result)
