"""
의학용어 RAG 검색 서비스
로컬(관리자 모드)과 Streamlit Cloud(일반 사용자 모드) 통합

모드 전환:
  - 로컬  : C:\Environment\.env 에 ADMIN_MODE=true 추가
  - Cloud : Streamlit Secrets에 ADMIN_MODE 미설정 (자동으로 일반 모드)
"""

import streamlit as st
from google import genai
from supabase import create_client
import os
from dotenv import load_dotenv
import urllib.parse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 환경변수 로드 ─────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=r"C:\Environment\.env")

def _get(key: str, default: str = "") -> str:
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)

# ── 모드 감지 ─────────────────────────────────────────────────────────────────
ADMIN_MODE = _get("ADMIN_MODE").lower() == "true"

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="의학용어 검색" + (" [관리자]" if ADMIN_MODE else ""),
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
        msg = ("⚠️ .env 파일에 SUPABASE_URL, SUPABASE_SERVICE_KEY, GEMINI_API_KEY를 설정하세요."
               if ADMIN_MODE else "서비스 설정 오류가 발생했습니다. 관리자에게 문의하세요.")
        st.error(msg)
        st.stop()

    gemini  = genai.Client(api_key=gemini_key)
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

# ── Gemini 한글 답변 생성 (병렬 호출) ────────────────────────────────────────
def _call_explanation(gemini_client, query: str, context: str) -> str:
    return gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""아래 의학 교재 내용만을 근거로 검색어에 대해 한국어로 정리하세요.
교재에 없는 내용은 추가하지 마세요.

검색어: {query}

--- 교재 내용 ---
{context}
---

다음 형식으로만 응답하세요. 대괄호 안을 실제 내용으로 채우세요.
응답에서 "교재", "교재에서", "교재는", "교재에 따르면" 등 교재를 직접 언급하는 표현은 절대 사용하지 마세요.

## [검색어의 영문 의학용어] / [한글 용어]

**정의**
[2~3문장으로 정의]

**핵심 개념**
[학습에 도움이 되는 핵심 포인트를 3~5개 제시하세요. 각 항목은 단순 나열이 아니라 왜 중요한지, 어떤 맥락에서 사용되는지 포함하여 2~3문장으로 설명하세요.]"""
    ).text.strip()

def _call_terms(gemini_client, query: str, context: str) -> str:
    return gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""아래 의학 교재 내용에서 의학용어를 추출하세요.
"{query}"는 제외하고 관련 의학용어만 추출하세요.

--- 교재 내용 ---
{context}
---

한 줄에 하나씩, "영문용어|한글용어" 형식으로만 응답하세요. 설명 없이 용어만."""
    ).text.strip()

def _call_meta(gemini_client, query: str, context: str) -> str:
    return gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""아래 의학 교재 내용을 바탕으로 "{query}"에 대한 약어와 분류를 추출하세요.

--- 교재 내용 ---
{context}
---

아래 형식으로만 응답하세요. 해당 정보가 없으면 항목을 비워두세요:
ABBREV: (약어, 없으면 빈칸)
CATEGORY: (의학 분류, 쉼표로 구분, 최대 3개)"""
    ).text.strip()

def _call_highlight(gemini_client, query: str, combined: str) -> str:
    return gemini_client.models.generate_content(
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

@st.cache_data(ttl=3600, show_spinner=False)
def generate_korean_answer(query: str, chunks_json: str, match_count: int = 10) -> tuple:
    """캐싱 적용: 같은 검색어는 1시간 내 즉시 반환"""
    import json
    chunks = json.loads(chunks_json)
    context = "\n\n".join([c.get("chunk_text", "") for c in chunks[:5]])

    # 1~3차 Gemini 호출 병렬 실행
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures["explanation"] = executor.submit(_call_explanation, gemini_client, query, context)
        futures["terms"]       = executor.submit(_call_terms,       gemini_client, query, context)
        futures["meta"]        = executor.submit(_call_meta,        gemini_client, query, context)
        if ADMIN_MODE and chunks:
            combined = "\n\n".join([c.get('chunk_text', '')[:400] for c in chunks[:5]])
            futures["highlight"] = executor.submit(_call_highlight, gemini_client, query, combined)

    explanation = futures["explanation"].result()
    terms_text  = futures["terms"].result()
    meta_text   = futures["meta"].result()

    # 관련 용어 파싱
    related_terms = []
    for line in terms_text.splitlines():
        line = line.strip().lstrip("-•1234567890. ")
        if "|" in line:
            parts = line.split("|", 1)
            en, ko = parts[0].strip(), parts[1].strip()
            if en and ko:
                related_terms.append({"en": en, "ko": ko})

    # 약어·분류 파싱
    abbrev = ""
    categories = []
    for line in meta_text.splitlines():
        line = line.strip()
        if line.startswith("ABBREV:"):
            abbrev = line.replace("ABBREV:", "").strip()
        elif line.startswith("CATEGORY:"):
            cats = line.replace("CATEGORY:", "").strip()
            categories = [c.strip() for c in cats.split(",") if c.strip()]

    # 원문 하이라이트 파싱 (관리자 모드에서만)
    highlighted_text = ""
    if ADMIN_MODE and "highlight" in futures:
        raw = futures["highlight"].result()
        ht = re.sub(r'\[SEARCH\](.*?)\[/SEARCH\]',
            r'<span style="background:#FFF3B0;color:#7A5C00;border-radius:3px;padding:1px 4px;">\1</span>', raw)
        ht = re.sub(r'\[PATH\](.*?)\[/PATH\]',
            r'<span style="background:#D4EDDA;color:#155724;border-radius:3px;padding:1px 4px;">\1</span>', ht)
        ht = re.sub(r'\[KEY\](.*?)\[/KEY\]',
            r'<span style="background:#CCE5FF;color:#004085;border-radius:3px;padding:1px 4px;">\1</span>', ht)
        highlighted_text = ht.replace("\n", "<br>")

    return explanation, related_terms[:8], abbrev, categories, highlighted_text

# ── KMA 공식 용어 조회 ────────────────────────────────────────────────────────
def kma_lookup(term: str) -> dict | None:
    try:
        result = supabase.table("kma_terms").select("en,ko").ilike("en", term).limit(1).execute()
        if result.data:
            return result.data[0]
        result = supabase.table("kma_terms").select("ko,en").ilike("ko", term).limit(1).execute()
        if result.data:
            return result.data[0]
    except Exception:
        pass
    return None

# ── 하이브리드 검색 ───────────────────────────────────────────────────────────
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_SOURCE_FILTER = "pdf"
DEFAULT_MATCH_COUNT   = 10

def hybrid_search(query: str, match_count: int = DEFAULT_MATCH_COUNT,
                  vector_weight: float = DEFAULT_VECTOR_WEIGHT,
                  source_filter: str = DEFAULT_SOURCE_FILTER):
    embedding = get_query_embedding(query)
    params = {
        "query_text":      query,
        "query_embedding": embedding,
        "match_count":     match_count,
        "vector_weight":   vector_weight,
        "fts_weight":      round(1.0 - vector_weight, 2),
        "source_filter":   source_filter,
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

# ── 관리자 모드: 사이드바 ─────────────────────────────────────────────────────
if ADMIN_MODE:
    with st.sidebar:
        st.header("⚙️ 검색 설정 (관리자)")
        match_count = st.slider("검색 결과 수", min_value=5, max_value=30,
                                value=DEFAULT_MATCH_COUNT, step=5)
        vector_weight = st.slider("벡터 검색 가중치", min_value=0.0, max_value=1.0,
                                  value=DEFAULT_VECTOR_WEIGHT, step=0.1,
                                  help="높을수록 의미 기반 검색 강화 (FTS 가중치 = 1 - 이 값)")
        source_filter = st.radio(
            "검색 소스",
            ["pdf", "all"],
            format_func=lambda x: {"all": "전체", "pdf": "교재 PDF만"}[x]
        )
        st.divider()
        st.markdown("""
        **검색 방식:** Hybrid Search (RRF)
        - 벡터 유사도 (코사인)
        - Full-Text Search (BM25)
        - Reciprocal Rank Fusion 결합
        """)
        st.caption(f"모드: {'🔴 관리자' if ADMIN_MODE else '🟢 일반'}")
else:
    match_count   = DEFAULT_MATCH_COUNT
    vector_weight = DEFAULT_VECTOR_WEIGHT
    source_filter = DEFAULT_SOURCE_FILTER

# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────
for key, val in [("sq", ""), ("explanation", ""), ("related_terms", []), ("results", []),
                 ("abbrev", ""), ("categories", []), ("no_result", False),
                 ("_last_q", ""), ("highlighted_text", ""), ("kma_term", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

def do_search(q: str):
    import json
    st.session_state.sq = q
    with st.spinner(f"🔍 '{q}' 검색 중..."):
        try:
            results = hybrid_search(q, match_count, vector_weight, source_filter)
            if not results:
                st.session_state.no_result = True
                st.session_state.explanation = ""
                st.session_state.related_terms = []
                st.session_state.results = []
                st.session_state.kma_term = None
            else:
                st.session_state.no_result = False
                st.session_state.results = results
                # KMA 조회와 Gemini 호출을 병렬 실행
                with ThreadPoolExecutor(max_workers=2) as executor:
                    chunks_json = json.dumps(results)
                    f_gemini = executor.submit(generate_korean_answer, q, chunks_json, match_count)
                    f_kma    = executor.submit(kma_lookup, q)
                    exp, terms, abbrev, categories, highlighted_text = f_gemini.result()
                    kma = f_kma.result()
                st.session_state.explanation     = exp
                st.session_state.related_terms   = terms
                st.session_state.abbrev          = abbrev
                st.session_state.categories      = categories
                st.session_state.highlighted_text = highlighted_text
                st.session_state.kma_term        = kma
        except Exception as e:
            err = str(e) if ADMIN_MODE else "검색 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            st.session_state.explanation   = err
            st.session_state.related_terms = []
            st.session_state.kma_term      = None

# ── URL query param 처리 ──────────────────────────────────────────────────────
if "q" in st.query_params:
    param_q = st.query_params["q"]
    if param_q and param_q != st.session_state._last_q:
        st.session_state._last_q = param_q
        do_search(param_q)
    del st.query_params["q"]

# ── 검색 탭 구성 ──────────────────────────────────────────────────────────────
if ADMIN_MODE:
    tab_search, tab_about, tab_stats = st.tabs(["🔍 검색", "ℹ️ 시스템 정보", "📊 통계"])
    search_container = tab_search
else:
    search_container = st.container()

with search_container:
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

    # ── 결과 표시 ──────────────────────────────────────────────────────────────
    if st.session_state.no_result:
        st.warning("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
    elif st.session_state.explanation:
        # KMA 공식 용어 배지
        if st.session_state.kma_term:
            kma = st.session_state.kma_term
            st.markdown(
                f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;'
                f'padding:10px 16px;margin-bottom:16px;display:flex;align-items:center;gap:10px;">'
                f'<span style="background:#1D4ED8;color:white;font-size:11px;font-weight:700;'
                f'padding:2px 8px;border-radius:4px;">의학용어집 제5판 기준</span>'
                f'<span style="font-size:14px;color:#1E3A8A;font-weight:600;">'
                f'{kma.get("en","")} &nbsp;·&nbsp; {kma.get("ko","")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(st.session_state.explanation)

        if st.session_state.abbrev or st.session_state.categories:
            badges_html = '<div style="margin:8px 0 16px 0;">'
            badges_html += '<span style="font-size:13px;color:#6B7280;margin-right:8px;">약어 / 분류</span>'
            if st.session_state.abbrev:
                badges_html += (f'<span style="background:#DBEAFE;color:#1E40AF;padding:4px 12px;'
                                f'border-radius:20px;font-size:13px;font-weight:500;margin:3px;">'
                                f'약어: {st.session_state.abbrev}</span>')
            for cat in st.session_state.categories:
                badges_html += (f'<span style="background:#D1FAE5;color:#065F46;padding:4px 12px;'
                                f'border-radius:20px;font-size:13px;font-weight:500;margin:3px;">'
                                f'{cat}</span>')
            badges_html += '</div>'
            st.markdown(badges_html, unsafe_allow_html=True)

        if st.session_state.related_terms:
            st.markdown("**관련 용어**")
            tags_html = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin:6px 0;">'
            for i, term in enumerate(st.session_state.related_terms[:8]):
                en    = urllib.parse.quote(term.get('en', ''))
                label = f"{term.get('en','')} / {term.get('ko','')}"
                tags_html += (f'<a href="?q={en}" target="_self" '
                              f'class="term-tag tag-{i % 8}">{label}</a>')
            tags_html += '</div>'
            st.markdown(tags_html, unsafe_allow_html=True)
            st.caption("💡 태그 클릭 시 해당 용어 바로 검색 가능")

        # 원문 보기 (관리자 모드에서만)
        if ADMIN_MODE and st.session_state.highlighted_text:
            with st.expander("📄 원문 보기", expanded=False):
                st.markdown(
                    f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;'
                    f'padding:1rem;font-size:14px;line-height:1.9;">'
                    f'{st.session_state.highlighted_text}</div>'
                    f'<div style="display:flex;gap:16px;margin-top:10px;font-size:12px;color:#6B7280;">'
                    f'<span><span style="background:#FFF3B0;color:#7A5C00;border-radius:3px;'
                    f'padding:1px 8px;margin-right:4px;">■</span>검색어</span>'
                    f'<span><span style="background:#D4EDDA;color:#155724;border-radius:3px;'
                    f'padding:1px 8px;margin-right:4px;">■</span>병태생리</span>'
                    f'<span><span style="background:#CCE5FF;color:#004085;border-radius:3px;'
                    f'padding:1px 8px;margin-right:4px;">■</span>핵심 용어</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # 검색 결과 청크 (관리자 모드에서만)
        if ADMIN_MODE and st.session_state.results:
            with st.expander(f"🗂️ 검색된 청크 ({len(st.session_state.results)}개)", expanded=False):
                for i, r in enumerate(st.session_state.results):
                    st.markdown(f"**#{i+1}** `score: {r.get('rrf_score', 0):.4f}` | "
                                f"`source: {r.get('source', '-')}` | "
                                f"`page: {r.get('page_num', '-')}`")
                    st.text(r.get('chunk_text', '')[:300])
                    st.divider()

# ── 관리자 전용 탭 ────────────────────────────────────────────────────────────
if ADMIN_MODE:
    with tab_about:
        st.markdown("""
        ### 시스템 구성

        | 구성요소 | 내용 |
        |---------|------|
        | **데이터** | 의학용어 교재 PDF (588쪽) + KMA 필수의학용어집 (23,291 용어) |
        | **임베딩 모델** | Gemini gemini-embedding-001 (3072차원) |
        | **벡터 DB** | Supabase PostgreSQL + pgvector |
        | **인덱스** | HNSW (m=16, ef_construction=64) |
        | **검색 방식** | Hybrid Search = 벡터(코사인) + FTS, RRF 결합 |
        | **LLM** | Gemini gemini-2.5-flash |
        | **프론트엔드** | Streamlit |
        | **배포** | Streamlit Cloud |

        ### 검색 파이프라인
        ```
        사용자 입력 → Gemini embed_content(RETRIEVAL_QUERY)
                    → Supabase hybrid_search RPC
                        ├─ 벡터 검색 (top-100, cosine similarity)
                        └─ Full-Text Search (top-100, tsvector)
                    → RRF 융합 (1/(k+rank), k=60)
                    → 상위 N개 반환
                    → Gemini gemini-2.5-flash (3~4회 호출)
                    → KMA 공식 용어 조회 (kma_terms 테이블)
        ```
        """)

    with tab_stats:
        if st.button("📊 DB 통계 조회"):
            with st.spinner("통계 수집 중..."):
                try:
                    pdf_count = supabase.table("mediterm_pdf_chunks").select("*", count="exact").execute()
                    kma_count = supabase.table("kma_terms").select("*", count="exact").execute()

                    col1, col2 = st.columns(2)
                    col1.metric("📘 PDF 청크 수", f"{pdf_count.count:,}")
                    col2.metric("🏥 KMA 용어 수", f"{kma_count.count:,}")
                except Exception as e:
                    st.error(f"통계 조회 오류: {e}")
