"""
의학용어 RAG 검색 서비스
Streamlit Cloud 배포용 (일반 사용자)
"""

import streamlit as st
from google import genai
from supabase import create_client
import os
from dotenv import load_dotenv
import urllib.parse
import re

# ── 환경변수 로드 ─────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=r"C:\Environment\.env")

def _get(key: str) -> str:
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key)

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="의학용어 검색",
    page_icon="🏥",
    layout="wide"
)

# ── CSS 스타일 ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4F46E5, #059669);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .term-tag {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 13px;
        font-weight: 500;
        margin: 4px;
        cursor: pointer;
        transition: opacity 0.15s;
    }
    .term-tag:hover { opacity: 0.8; }
    .tag-0 { background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE; }
    .tag-1 { background:#F0FDF4; color:#15803D; border:1px solid #86EFAC; }
    .tag-2 { background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; }
    .tag-3 { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
    .tag-4 { background:#F0F9FF; color:#075985; border:1px solid #7DD3FC; }
    .tag-5 { background:#FDF4FF; color:#6B21A8; border:1px solid #D8B4FE; }
    .tag-6 { background:#FFF7ED; color:#9A3412; border:1px solid #FDBA74; }
    .tag-7 { background:#F0FDFA; color:#134E4A; border:1px solid #99F6E4; }
</style>
""", unsafe_allow_html=True)

# ── 클라이언트 초기화 ─────────────────────────────────────────────────────────
@st.cache_resource
def init_clients():
    supabase_url = _get("SUPABASE_URL")
    supabase_key = _get("SUPABASE_SERVICE_KEY") or _get("SUPABASE_ANON_KEY")
    gemini_key   = _get("GEMINI_API_KEY")

    if not all([supabase_url, supabase_key, gemini_key]):
        st.error("서비스 설정 오류가 발생했습니다. 관리자에게 문의하세요.")
        st.stop()

    gemini = genai.Client(api_key=gemini_key)
    supabase = create_client(supabase_url, supabase_key)
    return supabase, gemini

supabase, gemini_client = init_clients()

# ── 임베딩 함수 ───────────────────────────────────────────────────────────────
def get_query_embedding(query: str) -> list[float]:
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config={"task_type": "RETRIEVAL_QUERY"}
    )
    return result.embeddings[0].values

# ── Gemini 한글 답변 생성 ──────────────────────────────────────────────────────
def generate_korean_answer(query: str, chunks: list) -> tuple:
    context = "\n\n".join([c.get("chunk_text", "") for c in chunks[:5]])

    # 1차: 한글 설명
    explanation = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""아래 의학 교재 내용만을 근거로 검색어에 대해 한국어로 정리하세요.
교재에 없는 내용은 추가하지 마세요.

검색어: {query}

--- 교재 내용 ---
{context}
---

다음 형식으로만 응답하세요. 대괄호 안을 실제 내용으로 채우세요:

## [검색어의 영문 의학용어] / [한글 용어]

**정의**
[교재 내용에 기반한 정의, 2~3문장]

**핵심 개념**
[교재 내용을 바탕으로 학습에 도움이 되는 핵심 포인트를 3~5개 제시하세요. 각 항목은 단순 나열이 아니라 왜 중요한지, 어떤 맥락에서 사용되는지 포함하여 2~3문장으로 설명하세요.]"""
    ).text.strip()

    # 2차: 관련 용어 추출
    terms_text = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""아래 의학 교재 내용에서 의학용어를 추출하세요.
"{query}"는 제외하고 관련 의학용어만 추출하세요.

--- 교재 내용 ---
{context}
---

한 줄에 하나씩, "영문용어|한글용어" 형식으로만 응답하세요. 설명 없이 용어만."""
    ).text.strip()

    related_terms = []
    for line in terms_text.splitlines():
        line = line.strip().lstrip("-•1234567890. ")
        if "|" in line:
            parts = line.split("|", 1)
            en, ko = parts[0].strip(), parts[1].strip()
            if en and ko:
                related_terms.append({"en": en, "ko": ko})

    # 3차: 약어 및 분류 추출
    meta_text = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""아래 의학 교재 내용을 바탕으로 "{query}"에 대한 약어와 분류를 추출하세요.

--- 교재 내용 ---
{context}
---

아래 형식으로만 응답하세요. 해당 정보가 없으면 항목을 비워두세요:
ABBREV: (약어, 없으면 빈칸)
CATEGORY: (의학 분류, 쉼표로 구분, 최대 3개)"""
    ).text.strip()

    abbrev = ""
    categories = []
    for line in meta_text.splitlines():
        line = line.strip()
        if line.startswith("ABBREV:"):
            abbrev = line.replace("ABBREV:", "").strip()
        elif line.startswith("CATEGORY:"):
            cats = line.replace("CATEGORY:", "").strip()
            categories = [c.strip() for c in cats.split(",") if c.strip()]

    # 4차: 원문 하이라이트
    highlighted_text = ""
    if chunks:
        combined = "\n\n".join([c.get('chunk_text', '')[:400] for c in chunks[:5]])
        raw = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""아래는 의학 교재에서 검색된 여러 단락입니다.
이 단락들을 자연스럽게 이어지는 하나의 영문 원문으로 재정리하세요.
중복 내용은 제거하고, 교재 원문의 표현과 문장을 최대한 그대로 유지하세요.

재정리 후, 아래 세 가지를 태그로 표시하세요:
1. [SEARCH]...[/SEARCH]: "{query}" 또는 그 동의어/약어
2. [PATH]...[/PATH]: 병태생리, 원인, 기전 관련 구절
3. [KEY]...[/KEY]: 다른 핵심 의학용어

교재 단락들:
{combined}

재정리된 단일 원문(태그 포함):"""
        ).text.strip()

        ht = re.sub(r'\[SEARCH\](.*?)\[/SEARCH\]',
            r'<span style="background:#FFF3B0;color:#7A5C00;border-radius:3px;padding:1px 4px;">\1</span>', raw)
        ht = re.sub(r'\[PATH\](.*?)\[/PATH\]',
            r'<span style="background:#D4EDDA;color:#155724;border-radius:3px;padding:1px 4px;">\1</span>', ht)
        ht = re.sub(r'\[KEY\](.*?)\[/KEY\]',
            r'<span style="background:#CCE5FF;color:#004085;border-radius:3px;padding:1px 4px;">\1</span>', ht)
        highlighted_text = ht.replace("\n", "<br>")

    return explanation, related_terms[:8], abbrev, categories, highlighted_text

# ── KMA 공식 용어 조회 ───────────────────────────────────────────────────────
def kma_lookup(term: str) -> dict | None:
    """대한의사협회 필수의학용어집에서 공식 한글/영어 용어 조회"""
    try:
        # 영어로 조회
        result = supabase.table("kma_terms") \
            .select("en,ko") \
            .ilike("en", term) \
            .limit(1).execute()
        if result.data:
            return result.data[0]
        # 한글로 조회
        result = supabase.table("kma_terms") \
            .select("ko,en") \
            .ilike("ko", term) \
            .limit(1).execute()
        if result.data:
            return result.data[0]
    except Exception:
        pass
    return None


# ── 하이브리드 검색 (고정 설정) ───────────────────────────────────────────────
VECTOR_WEIGHT = 0.7
SOURCE_FILTER = "pdf"
MATCH_COUNT   = 10

def hybrid_search(query: str):
    embedding = get_query_embedding(query)
    params = {
        "query_text":      query,
        "query_embedding": embedding,
        "match_count":     MATCH_COUNT,
        "vector_weight":   VECTOR_WEIGHT,
        "fts_weight":      round(1.0 - VECTOR_WEIGHT, 2),
        "source_filter":   SOURCE_FILTER,
    }
    response = supabase.rpc("hybrid_search", params).execute()
    return response.data

# ── UI 헤더 ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 의학용어 검색</h1>
    <p>AI 기반 의학용어 검색 서비스</p>
    <small>한국어 · 영어 모두 검색 가능합니다</small>
</div>
""", unsafe_allow_html=True)

# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────
for key, val in [("sq", ""), ("explanation", ""), ("related_terms", []), ("results", []),
                 ("abbrev", ""), ("categories", []), ("no_result", False),
                 ("_last_q", ""), ("highlighted_text", ""), ("kma_term", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

def do_search(q: str):
    st.session_state.sq = q
    with st.spinner(f"🔍 '{q}' 검색 중..."):
        try:
            results = hybrid_search(q)
            if not results:
                st.session_state.explanation = ""
                st.session_state.related_terms = []
                st.session_state.results = []
                st.session_state.no_result = True
                st.session_state.kma_term = None
            else:
                st.session_state.no_result = False
                st.session_state.results = results
                exp, terms, abbrev, categories, highlighted_text = generate_korean_answer(q, results)
                st.session_state.explanation = exp
                st.session_state.related_terms = terms
                st.session_state.abbrev = abbrev
                st.session_state.categories = categories
                st.session_state.highlighted_text = highlighted_text
                # KMA 공식 용어 조회
                st.session_state.kma_term = kma_lookup(q)
        except Exception as e:
            st.session_state.explanation = "검색 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            st.session_state.related_terms = []
            st.session_state.kma_term = None

# ── URL query param으로 용어 태그 클릭 처리 ──────────────────────────────────
if "q" in st.query_params:
    param_q = st.query_params["q"]
    if param_q and param_q != st.session_state._last_q:
        st.session_state._last_q = param_q
        do_search(param_q)
    del st.query_params["q"]

# ── 검색 UI ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "검색어 입력",
        placeholder="예: 당뇨병, hypertension, 심근경색, pneumonia...",
        label_visibility="collapsed",
        value=st.session_state.sq
    )
with col2:
    search_btn = st.button("검색", type="primary", use_container_width=True)

st.caption("예시 검색어:")
ex_cols = st.columns(5)
examples = ["당뇨병", "hypertension", "심근경색", "폐렴", "골절"]
for i, ex in enumerate(examples):
    if ex_cols[i].button(ex, key=f"ex_{i}"):
        do_search(ex)

if search_btn and query.strip():
    do_search(query)

# ── 결과 표시 ─────────────────────────────────────────────────────────────────
if st.session_state.no_result:
    st.warning("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
elif st.session_state.explanation:
    # KMA 공식 용어 배지
    if st.session_state.kma_term:
        kma = st.session_state.kma_term
        en_disp = kma.get('en', '')
        ko_disp = kma.get('ko', '')
        st.markdown(
            f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;'
            f'padding:10px 16px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">'
            f'<span style="background:#1D4ED8;color:white;font-size:11px;font-weight:700;'
            f'padding:2px 8px;border-radius:4px;">대한의사협회 공식</span>'
            f'<span style="font-size:14px;color:#1E3A8A;font-weight:600;">'
            f'{en_disp} &nbsp;·&nbsp; {ko_disp}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown(st.session_state.explanation)

    if st.session_state.abbrev or st.session_state.categories:
        badges_html = '<div style="margin:8px 0 16px 0;">'
        badges_html += '<span style="font-size:13px;color:#6B7280;margin-right:8px;">약어 / 분류</span>'
        if st.session_state.abbrev:
            badges_html += f'<span style="background:#DBEAFE;color:#1E40AF;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:500;margin:3px;">약어: {st.session_state.abbrev}</span>'
        for cat in st.session_state.categories:
            badges_html += f'<span style="background:#D1FAE5;color:#065F46;padding:4px 12px;border-radius:20px;font-size:13px;font-weight:500;margin:3px;">{cat}</span>'
        badges_html += '</div>'
        st.markdown(badges_html, unsafe_allow_html=True)

    if st.session_state.related_terms:
        st.markdown("**관련 용어**")
        tags_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:6px 0;">'
        for i, term in enumerate(st.session_state.related_terms[:8]):
            en = urllib.parse.quote(term.get('en', ''))
            label = f"{term.get('en', '')} / {term.get('ko', '')}"
            tags_html += f'<a href="?q={en}" target="_self" class="term-tag tag-{i % 8}">{label}</a>'
        tags_html += '</div>'
        st.markdown(tags_html, unsafe_allow_html=True)
        st.caption("💡 태그 클릭 시 해당 용어 바로 검색 가능")

