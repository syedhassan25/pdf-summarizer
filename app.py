import os
import re
import time
import tempfile

import pymupdf as fitz  # PyMuPDF
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # production par .env nahi hoti
    pass

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="ResearchAI | Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Settings ----------------
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
TIER = os.getenv("GROQ_TIER", "free")  # "free" or "paid"
MAX_PDF_MB = 20
MAX_PAGES = int(os.getenv("MAX_PAGES", "60"))
STRIP_REFERENCES = True

PRESETS = {
    "free": dict(SINGLE_PASS_LIMIT=14_000, CHUNK_SIZE=12_000, CHUNK_OVERLAP=500,
                 MAX_CONCURRENCY=1, REDUCE_FAN_IN=5, MAP_MAX_TOKENS=1500, REPORT_MAX_TOKENS=3000),
    "paid": dict(SINGLE_PASS_LIMIT=300_000, CHUNK_SIZE=80_000, CHUNK_OVERLAP=2_000,
                 MAX_CONCURRENCY=4, REDUCE_FAN_IN=8, MAP_MAX_TOKENS=3000, REPORT_MAX_TOKENS=6000),
}
if TIER not in PRESETS:
    raise ValueError(f"GROQ_TIER must be one of {list(PRESETS)}, got {TIER!r}")
CFG = PRESETS[TIER]


# ---------------- Models ----------------
@st.cache_resource
def make_llm(max_tokens: int) -> ChatGroq:
    # Streamlit Secrets se ya environment variable se key uthayega
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
            
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it as an environment variable or in Streamlit Secrets."
        )
    common = dict(model=MODEL_NAME, api_key=api_key, temperature=0.2,
                  max_tokens=max_tokens, max_retries=6, timeout=120)
    try:
        return ChatGroq(reasoning_effort="low", **common)
    except Exception as exc:
        print("reasoning_effort not accepted, using default:", exc)
        return ChatGroq(**common)


llm_map = make_llm(CFG["MAP_MAX_TOKENS"])
llm_report = make_llm(CFG["REPORT_MAX_TOKENS"])


# ---------------- Helpers ----------------
def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts)
    return str(content)


def strip_references(text: str) -> str:
    matches = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text, flags=re.IGNORECASE))
    for m in reversed(matches):
        if m.start() > len(text) * 0.6:
            return text[:m.start()]
    return text


def extract_pdf_text(pdf_path):
    if not pdf_path or not os.path.isfile(pdf_path):
        raise ValueError("Please upload a valid PDF file.")
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        raise ValueError(f"PDF is {size_mb:.1f} MB; the limit is {MAX_PDF_MB} MB.")

    pages_text = []
    with fitz.open(pdf_path) as document:
        if len(document) == 0:
            raise ValueError("The uploaded PDF has no pages.")
        if len(document) > MAX_PAGES:
            raise ValueError(f"PDF has {len(document)} pages; the limit is {MAX_PAGES}.")
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                pages_text.append(f"\n[PAGE {page_number}]\n{page_text}")

    full_text = "\n".join(pages_text).strip()
    if not full_text:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; "
            "OCR support is not included in this version."
        )
    if STRIP_REFERENCES:
        full_text = strip_references(full_text)
    return full_text, len(pages_text)


def split_document(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG["CHUNK_SIZE"],
        chunk_overlap=CFG["CHUNK_OVERLAP"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def run_parallel(prompts):
    responses = llm_map.batch(prompts, config={"max_concurrency": CFG["MAX_CONCURRENCY"]})
    return [extract_text_content(r.content) for r in responses]


# ---------------- Prompts ----------------
def map_prompt(chunk, index, total):
    return f"""You are an expert academic research analyst.
Summarize section {index} of {total} from a research paper in under 300 words.

Extract only information supported by the text:
- Main ideas and research problem
- Objectives and methodology, if present
- Findings, evidence, and numerical results
- Limitations, if stated
- Important technical terms

Do not invent missing information.
This is one section, not necessarily the whole paper.

SECTION:
{chunk}
"""


def reduce_prompt(batch):
    combined = "\n\n---\n\n".join(batch)
    return f"""Combine these partial research-paper summaries into
one accurate intermediate summary of under 300 words.

Merge overlapping points and remove repetition. Preserve methods,
important findings, numbers, and limitations. Do not add unsupported facts.

PARTIAL SUMMARIES:
{combined}
"""


def stream_final_report(source_text):
    prompt = f"""You are a senior academic research analyst.
Create a coherent report based only on the research information below.

Use these headings:
# Executive Summary
# Research Problem
# Research Objectives
# Methodology
# Key Findings
# Important Technical Concepts
# Limitations
# Conclusion

If a heading is not supported, say it is not clearly specified in the
provided text. Do not fabricate citations, results, or numerical values.

RESEARCH INFORMATION:
{source_text}
"""
    report = ""
    for piece in llm_report.stream(prompt):
        report += extract_text_content(piece.content)
        yield report


# ---------------- Core Pipeline Logic ----------------
def analyze_paper(pdf_path, status_text_placeholder):
    start = time.time()

    def elapsed() -> str:
        return f"{time.time() - start:.0f}s"

    try:
        status_text_placeholder.text("Extracting PDF text...")
        text, pages_with_text = extract_pdf_text(pdf_path)
        status_text_placeholder.text(f"Extracted {len(text):,} characters from {pages_with_text} pages. Calling Groq...")

        if len(text) <= CFG["SINGLE_PASS_LIMIT"]:
            source, n_calls = text, 1
        else:
            chunks = split_document(text)
            step = CFG["MAX_CONCURRENCY"]
            partials = []
            for i in range(0, len(chunks), step):
                group = chunks[i:i + step]
                prompts = [map_prompt(c, i + j + 1, len(chunks)) for j, c in enumerate(group)]
                partials += run_parallel(prompts)
                status_text_placeholder.text(f"Analyzed {len(partials)}/{len(chunks)} sections... ({elapsed()})")

            n_calls = len(chunks) + 1
            current = partials
            while len(current) > 1:
                fan = CFG["REDUCE_FAN_IN"]
                batches = [current[k:k + fan] for k in range(0, len(current), fan)]
                status_text_placeholder.text(f"Combining summaries ({len(current)} -> {len(batches)})... ({elapsed()})")
                current = run_parallel([reduce_prompt(b) for b in batches])
                n_calls += len(batches)
            source = current[0]

        status_text_placeholder.text(f"Writing final report... ({elapsed()})")
        report = ""
        # Stream output inside the report area dynamically if possible, or fetch final
        for partial_report in stream_final_report(source):
            report = partial_report

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="research_summary_",
            delete=False, encoding="utf-8",
        ) as f:
            f.write(report)
            output_path = f.name

        status_text_placeholder.text(f"Complete in {elapsed()} · {pages_with_text} pages · {n_calls} API call(s)")
        return report, output_path

    except Exception as exc:
        raise RuntimeError(f"Could not analyze this PDF: {exc}")


# ---------------- Streamlit UI ----------------
st.markdown("<h1 style='text-align: center;'>📚 ResearchAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.7;'>Turn lengthy academic papers into structured insights.</p>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    st.subheader("Upload & Settings")
    uploaded_file = st.file_uploader("Upload research paper", type=[".pdf"])
    
    st.info(f"Upload a text-based PDF (max {MAX_PAGES} pages, {MAX_PDF_MB} MB). Scanned PDFs are not supported.")
    
    with st.expander("What you'll receive"):
        st.markdown(
            "- Executive summary\n"
            "- Research problem and objectives\n"
            "- Methodology and key findings\n"
            "- Technical concepts and limitations\n"
            "- Conclusion"
        )
    
    analyze_btn = st.button("✨ Analyze Research Paper", type="primary", use_container_width=True)

with col2:
    st.subheader("Analysis Report")
    status_placeholder = st.empty()
    report_placeholder = st.empty()
    download_placeholder = st.empty()
    
    report_placeholder.markdown("Your report will appear here after analysis.")

if analyze_btn:
    if not uploaded_file:
        st.error("Please upload a research paper PDF first.")
    else:
        # Save uploaded file temporarily to disk so pymupdf can read it easily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Processing paper through pipeline..."):
                report_text, md_file_path = analyze_paper(tmp_path, status_placeholder)
                
                # Display output report
                report_placeholder.markdown(report_text)
                
                # Setup download button for the markdown report
                with open(md_file_path, "r", encoding="utf-8") as f:
                    md_bytes = f.read()
                
                download_placeholder.download_button(
                    label="📥 Download Summary (.md)",
                    data=md_bytes,
                    file_name="research_summary.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        except Exception as e:
            st.error(str(e))
        finally:
            # Clean up temp input pdf
            if os.path.exists(tmp_path):
                os.remove(tmp_path)