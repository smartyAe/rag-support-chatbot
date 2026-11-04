"""Web chat: uvicorn docs_assistant.app:app --app-dir src --port 8000

GET  /            chat page (append ?q=... to pre-render questions)
POST /api/ask     {"question": "..."} -> {"answer", "sources", "mode", "answered"}
"""

from __future__ import annotations

import html
import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .answer import Answer, Assistant

DOCS = Path(os.getenv("DOCS_FOLDER", Path(__file__).resolve().parents[2] / "docs_kb"))
assistant = Assistant(DOCS)
app = FastAPI(title="Docs assistant")


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)


def _to_json(answer: Answer) -> dict:
    return {
        "answer": answer.text,
        "answered": answer.answered,
        "mode": answer.mode,
        "sources": [{"title": c.title, "section": c.section, "file": c.file, "excerpt": c.text}
                    for c in answer.sources],
    }


@app.post("/api/ask")
def ask(body: Question) -> dict:
    return _to_json(assistant.ask(body.question))


def _bubble(question: str, answer: Answer) -> str:
    sources = "".join(
        f'<details><summary>[{i}] {html.escape(c.citation)}</summary>'
        f'<p>{html.escape(c.text)}</p></details>' for i, c in enumerate(answer.sources, 1))
    klass = "bot" if answer.answered else "bot unknown"
    return (f'<div class="msg user">{html.escape(question)}</div>'
            f'<div class="msg {klass}"><p>{html.escape(answer.text)}</p>{sources}</div>')


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kora support assistant</title>
<style>
  :root { --navy:#0f3d5e; --bg:#f4f7fa; --line:#d8e1ea; }
  * { box-sizing:border-box; }
  body { margin:0; font:16px/1.5 "Segoe UI", system-ui, sans-serif; background:var(--bg); color:#1e293b; }
  header { background:var(--navy); color:#fff; padding:18px 28px; }
  header h1 { margin:0; font-size:20px; }
  header p { margin:2px 0 0; opacity:.8; font-size:14px; }
  main { max-width:860px; margin:0 auto; padding:24px 16px 120px; }
  .msg { padding:12px 16px; border-radius:14px; margin:10px 0; max-width:85%; width:fit-content; }
  .user { background:var(--navy); color:#fff; margin-left:auto; border-bottom-right-radius:4px; }
  .bot { background:#fff; border:1px solid var(--line); border-bottom-left-radius:4px; }
  .bot p { margin:0 0 6px; }
  .unknown { border-color:#f2c48d; background:#fffaf2; }
  details { font-size:13px; color:#475569; border-top:1px dashed var(--line); padding-top:6px; margin-top:6px; }
  summary { cursor:pointer; color:var(--navy); font-weight:600; }
  form { position:fixed; bottom:0; left:0; right:0; background:#fff; border-top:1px solid var(--line); padding:14px 16px; }
  .row { max-width:860px; margin:0 auto; display:flex; gap:10px; }
  input { flex:1; padding:12px 14px; border:1px solid var(--line); border-radius:10px; font-size:16px; }
  button { background:var(--navy); color:#fff; border:0; border-radius:10px; padding:0 20px; font-size:16px; cursor:pointer; }
  .mode { font-size:12px; color:#64748b; text-align:center; }
</style></head>
<body>
<header><h1>Kora support assistant</h1>
<p>Answers come only from the company documentation, with their sources.</p></header>
<main id="chat">__MESSAGES__<p class="mode">Answer mode: __MODE__</p></main>
<form id="f"><div class="row"><input id="q" placeholder="Ask about delivery, returns, warranty, invoices..." autocomplete="off">
<button>Ask</button></div></form>
<script>
const chat = document.getElementById("chat"), q = document.getElementById("q");
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
document.getElementById("f").addEventListener("submit", async e => {
  e.preventDefault();
  const question = q.value.trim(); if (!question) return;
  q.value = "";
  chat.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(question)}</div>`);
  const r = await fetch("/api/ask", {method:"POST", headers:{"Content-Type":"application/json"},
                                     body: JSON.stringify({question})});
  const a = await r.json();
  const src = a.sources.map((s, i) => `<details><summary>[${i+1}] ${esc(s.title)} › ${esc(s.section)}</summary><p>${esc(s.excerpt)}</p></details>`).join("");
  chat.insertAdjacentHTML("beforeend", `<div class="msg bot ${a.answered ? "" : "unknown"}"><p>${esc(a.answer)}</p>${src}</div>`);
  window.scrollTo(0, document.body.scrollHeight);
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page(q: list[str] = Query(default=[])) -> str:
    messages = "".join(_bubble(question, assistant.ask(question)) for question in q[:6])
    mode = ("OpenAI model " + assistant.model) if assistant.use_llm else "extractive (offline, no API key)"
    return PAGE.replace("__MESSAGES__", messages).replace("__MODE__", html.escape(mode))
