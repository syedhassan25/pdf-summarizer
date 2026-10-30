"""ResearchAI - research paper summarizer (Gradio + Groq). Deployable on Render / any Python host."""
import os
import re
import time
import tempfile

import pymupdf as fitz  # PyMuPDF
import gradio as gr
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # production par .env nahi hoti
    pass

# ---------------- Settings ----------------
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
TIER = os.getenv("GROQ_TIER", "free")  # "free" or "paid" (set as a Space variable)
MAX_PDF_MB = 20
MAX_PAGES = int(os.getenv("MAX_PAGES", "60"))
STRIP_REFERENCES = True

PRESETS = {
    # Each request (input + output tokens) must stay under the per-minute token cap.
    "free": dict(SINGLE_PASS_LIMIT=14_000, CHUNK_SIZE=12_000, CHUNK_OVERLAP=500,
                 MAX_CONCURRENCY=1, REDUCE_FAN_IN=5, MAP_MAX_TOKENS=1500, REPORT_MAX_TOKENS=3000),
    "paid": dict(SINGLE_PASS_LIMIT=300_000, CHUNK_SIZE=80_000, CHUNK_OVERLAP=2_000,
                 MAX_CONCURRENCY=4, REDUCE_FAN_IN=8, MAP_MAX_TOKENS=3000, REPORT_MAX_TOKENS=6000),
}
if TIER not in PRESETS:
    raise ValueError(f"GROQ_TIER must be one of {list(PRESETS)}, got {TIER!r}")
CFG = PRESETS[TIER]


# ---------------- Models ----------------
def make_llm(max_tokens: int) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it as an environment variable in your hosting dashboard."
        )
    common = dict(model=MODEL_NAME, api_key=api_key, temperature=0.2,
                  max_tokens=max_tokens, max_retries=6, timeout=120)
    try:
        return ChatGroq(reasoning_effort="low", **common)  # gpt-oss: faster, fewer tokens
    except Exception as exc:  # older langchain-groq without reasoning_effort
        print("reasoning_effort not accepted, using default:", exc)
        return ChatGroq(**common)


llm_map = make_llm(CFG["MAP_MAX_TOKENS"])
llm_report = make_llm(CFG["REPORT_MAX_TOKENS"])


# ---------------- Helpers ----------------
def extract_text_content(content) -> str:
    """Convert LLM response content to a clean string."""
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
    """Cut the bibliography if a 'References' heading appears in the last 40% of the text."""
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


# ---------------- Main entry point ----------------
def analyze_paper(pdf_path):
    if not pdf_path:
        raise gr.Error("Upload a research paper PDF first.")
    start = time.time()

    def elapsed() -> str:
        return f"{time.time() - start:.0f}s"

    try:
        yield "Extracting PDF text...", "", None
        text, pages_with_text = extract_pdf_text(pdf_path)
        yield (f"Extracted {len(text):,} characters from {pages_with_text} pages. "
               f"Calling Groq..."), "", None

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
                yield f"Analyzed {len(partials)}/{len(chunks)} sections... ({elapsed()})", "", None

            n_calls = len(chunks) + 1
            current = partials
            while len(current) > 1:
                fan = CFG["REDUCE_FAN_IN"]
                batches = [current[k:k + fan] for k in range(0, len(current), fan)]
                yield f"Combining summaries ({len(current)} -> {len(batches)})... ({elapsed()})", "", None
                current = run_parallel([reduce_prompt(b) for b in batches])
                n_calls += len(batches)
            source = current[0]

        report = ""
        for report in stream_final_report(source):
            yield f"Writing report... ({elapsed()})", report, None

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="research_summary_",
            delete=False, encoding="utf-8",
        ) as f:
            f.write(report)
            output_path = f.name

        yield (f"**Complete** in {elapsed()} · {pages_with_text} pages · "
               f"{n_calls} API call(s)"), report, output_path

    except gr.Error:
        raise
    except Exception as exc:
        print(f"analyze_paper failed: {exc!r}")  # full detail goes to the Space logs
        raise gr.Error(f"Could not analyze this PDF: {exc}") from exc


# ---------------- UI ----------------
CSS = """
<style>
.gradio-container {max-width: 1100px !important; margin: auto !important;}
#hero {text-align: center; padding: 22px 10px 12px;}
#hero h1 {font-size: 2.2rem; margin-bottom: 6px;}
#hero p {opacity: .7; font-size: 1rem;}
</style>
"""

with gr.Blocks(title="ResearchAI | Paper Summarizer") as demo:
    gr.HTML(CSS + """
    <div id="hero">
      <h1>📚 ResearchAI</h1>
      <p>Turn lengthy academic papers into structured insights.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload research paper", file_types=[".pdf"], type="filepath")
            analyze_button = gr.Button("✨ Analyze Research Paper", variant="primary", size="lg")
            status = gr.Markdown(
                f"Upload a text-based PDF (max {MAX_PAGES} pages, {MAX_PDF_MB} MB). "
                "Scanned PDFs are not supported."
            )
            gr.Markdown(
                "**What you'll receive**\n"
                "- Executive summary\n"
                "- Research problem and objectives\n"
                "- Methodology and key findings\n"
                "- Technical concepts and limitations\n"
                "- Conclusion"
            )
        with gr.Column(scale=2):
            report_output = gr.Markdown(value="Your report will appear here after analysis.")
            download_output = gr.File(label="Download summary (.md)")

    analyze_button.click(
        fn=analyze_paper,
        inputs=[pdf_input],
        outputs=[status, report_output, download_output],
    )

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    # Render (and most hosts) provide the port via $PORT and need 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))