"""Measure the retrieval quality on a fixed set of questions.

- hit@1 / hit@3: the expected section is the first / among the first 3 results
- refusals: out-of-scope questions must get "I couldn't find this", not an answer
- false refusals: in-scope questions that were wrongly refused

Usage: python eval/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docs_assistant.answer import Assistant  # noqa: E402


def evaluate(verbose: bool = True) -> dict:
    assistant = Assistant(ROOT / "docs_kb", use_llm=False)
    cases = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    in_scope = [c for c in cases if c["section"]]
    out_scope = [c for c in cases if not c["section"]]

    hit1 = hit3 = false_refusals = 0
    for case in in_scope:
        sections = [h.chunk.section for h in assistant.retrieve(case["q"])]
        hit1 += bool(sections) and sections[0] == case["section"]
        hit3 += case["section"] in sections
        false_refusals += not sections
        if verbose and (not sections or sections[0] != case["section"]):
            print(f"  miss: {case['q']!r} -> {sections[:3]} (expected {case['section']!r})")

    refusals = sum(not assistant.retrieve(c["q"]) for c in out_scope)
    for case in out_scope:
        if verbose and assistant.retrieve(case["q"]):
            print(f"  answered out-of-scope: {case['q']!r}")

    result = {
        "questions": len(cases),
        "hit@1": hit1 / len(in_scope),
        "hit@3": hit3 / len(in_scope),
        "false_refusals": false_refusals,
        "out_of_scope_refused": f"{refusals}/{len(out_scope)}",
    }
    if verbose:
        print(f"{len(in_scope)} in-scope questions: hit@1 {result['hit@1']:.0%}, "
              f"hit@3 {result['hit@3']:.0%}, wrongly refused {false_refusals}")
        print(f"{len(out_scope)} out-of-scope questions refused: {refusals}/{len(out_scope)}")
    return result


if __name__ == "__main__":
    evaluate()
