---
title: ResearchAI Paper Summarizer
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# ResearchAI

Upload a text-based research paper PDF and get a structured summary
(executive summary, problem, objectives, methodology, findings, limitations, conclusion).

Powered by Groq (`openai/gpt-oss-20b`), LangChain and PyMuPDF.

**Setup:** add a secret named `GROQ_API_KEY` in Space Settings -> Variables and secrets.
Optional variable: `GROQ_TIER` = `free` (default) or `paid`.
