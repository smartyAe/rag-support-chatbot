"""BM25 keyword search over the chunks (no external service, no embeddings bill).

BM25 is the ranking function used by Elasticsearch and Lucene. For a small
knowledge base it is fast, explainable and free; it can be swapped for vector
search (OpenAI embeddings, pgvector, Chroma...) behind the same interface.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .ingest import Chunk

STOPWORDS = set("""
a an and are as at be but by can do does for from has have how i if in into is it its
me my of on or our so than that the their them then there these they this to too us
was we what when where which who why will with you your yours am been being did doing
any all also about after before should would could may might must not no just only
""".split())


def stem(word: str) -> str:
    """Very light English stemmer: invoices/invoice, returned/return, refunds/refund."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
                word = word[:-1]  # shipping -> shipp -> ship
            break
    if word.endswith("y") and len(word) > 4:
        word = word[:-1]  # delivery / delivered -> deliver
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


# Words customers use -> words the documentation uses. This glossary is the
# part to tune for each client; it is applied to questions only.
SYNONYMS = {
    "package": "parcel", "box": "parcel",
    "broken": "damaged", "broke": "damaged",
    "money back": "refund", "reimburse": "refund", "reimbursement": "refund",
    "send back": "return", "send something back": "return",
    "swap": "exchange", "replace": "exchange",
    "ship": "delivery", "shipping": "delivery",
    "spilled": "liquid", "spill": "liquid", "coffee": "liquid", "water": "liquid",
    "free": "cost", "price": "cost", "how much": "cost",
    "install": "installation", "assemble": "installation",
    "late": "late reminder",
    "colleague": "user", "coworker": "user",
}


def expand(question: str) -> str:
    q = question.lower()
    extra = [target for phrase, target in SYNONYMS.items()
             if re.search(rf"\b{re.escape(phrase)}\b", q)]
    return question + (" " + " ".join(extra) if extra else "")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text) if w not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    coverage: float  # share of the question's words found in the chunk


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        # The title and section name are indexed too: they often hold the key word.
        self.docs = [Counter(tokenize(f"{c.title} {c.section} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, question: str, k: int = 3) -> list[Hit]:
        asked = set(tokenize(question))
        terms = list(dict.fromkeys(tokenize(expand(question))))
        known = [t for t in terms if t in self.idf]
        if not known:
            return []
        hits = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths):
            score = 0.0
            for t in known:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                    score += self.idf[t] * norm
            if score > 0:
                # A question word counts as covered if it, or its glossary
                # translation, appears in the chunk.
                found = {t for t in terms if t in doc}
                expanded_hit = bool(found - asked)
                coverage = min(1.0, (len(found & asked) + expanded_hit) / max(len(asked), 1))
                hits.append(Hit(chunk, score, coverage))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
