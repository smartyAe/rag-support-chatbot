"""Turn retrieved chunks into an answer that always cites its sources.

Two modes behind one interface:
- "llm": an OpenAI chat model writes the answer from the retrieved chunks only
  (opt-in: ASSISTANT_MODE=llm, plus OPENAI_API_KEY);
- "extractive": no API key needed, the best sentences of the best chunks are
  returned as-is. Useful offline, in tests, and as a fallback.

In both modes the assistant says it does not know when the documents do not
cover the question, instead of making something up.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest import Chunk, load_folder
from .retrieve import BM25Index, Hit, expand, tokenize

log = logging.getLogger(__name__)

# Below these values the best chunk is considered unrelated to the question.
MIN_SCORE = 3.0
MIN_COVERAGE = 0.5

NO_ANSWER = ("I couldn't find this in the documentation. "
             "Please contact the support team, who will be able to help.")

SYSTEM_PROMPT = """You are the customer support assistant of Kora Office Supplies.
Answer ONLY with information from the numbered sources below.
Cite the sources you used like [1] or [2]. Keep the answer short (2-4 sentences).
If the sources do not contain the answer, reply exactly: "{no_answer}"
Never invent prices, delays or policies."""


@dataclass
class Answer:
    text: str
    sources: list[Chunk] = field(default_factory=list)
    mode: str = "extractive"
    answered: bool = True


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class Assistant:
    def __init__(self, docs_folder: Path, use_llm: bool | None = None, model: str | None = None):
        self.chunks = load_folder(docs_folder)
        self.index = BM25Index(self.chunks)
        # The LLM mode is opt-in: an API key lying in the environment is not
        # enough, so the assistant never calls a paid API by surprise.
        self.use_llm = os.getenv("ASSISTANT_MODE", "").lower() == "llm" if use_llm is None else use_llm
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def retrieve(self, question: str, k: int = 3) -> list[Hit]:
        hits = self.index.search(question, k)
        if not hits or hits[0].score < MIN_SCORE or hits[0].coverage < MIN_COVERAGE:
            return []
        return hits

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits:
            return Answer(NO_ANSWER, [], "llm" if self.use_llm else "extractive", answered=False)
        if not self.use_llm:
            return self._extractive(question, hits)
        try:
            return self._llm(question, hits)
        except Exception as exc:  # API down, quota, wrong key: still answer from the docs
            log.warning("LLM call failed (%s), falling back to extractive answer", type(exc).__name__)
            return self._extractive(question, hits)

    def _extractive(self, question: str, hits: list[Hit]) -> Answer:
        asked = set(tokenize(question))
        terms = set(tokenize(expand(question)))
        best = hits[0].chunk
        scored = []
        for pos, sentence in enumerate(_sentences(best.text)):
            # The customer's own words count double; glossary words help break ties.
            overlap = sum(self.index.idf.get(t, 0) * (2 if t in asked else 1)
                          for t in set(tokenize(sentence)) & terms)
            scored.append((overlap, pos, sentence))
        top = sorted(scored, key=lambda s: (-s[0], s[1]))[:2]  # ties: earlier sentence first
        top = [s for s in top if s[0] > 0] or top[:1]
        text = " ".join(s for _, _, s in sorted(top, key=lambda s: s[1]))
        return Answer(f"{text} [1]", [best], "extractive")

    def _llm(self, question: str, hits: list[Hit]) -> Answer:
        from openai import OpenAI  # imported here so the offline mode needs no key

        context = "\n\n".join(f"[{i}] {h.chunk.citation}\n{h.chunk.text}"
                              for i, h in enumerate(hits, start=1))
        response = OpenAI().chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(no_answer=NO_ANSWER)},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
            ],
        )
        text = response.choices[0].message.content.strip()
        if text.startswith(NO_ANSWER[:30]):
            return Answer(NO_ANSWER, [], "llm", answered=False)
        cited = sorted({int(n) for n in re.findall(r"\[(\d)\]", text) if 0 < int(n) <= len(hits)})
        sources = [hits[n - 1].chunk for n in cited] or [hits[0].chunk]
        return Answer(text, sources, "llm")
