"""Terminal chat: python -m docs_assistant [question]"""

import sys
from pathlib import Path

from .answer import Assistant

assistant = Assistant(Path(__file__).resolve().parents[2] / "docs_kb")


def show(question: str) -> None:
    answer = assistant.ask(question)
    print(f"\n{answer.text}")
    for i, chunk in enumerate(answer.sources, 1):
        print(f"  [{i}] {chunk.citation}  ({chunk.file})")


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    mode = f"OpenAI ({assistant.model})" if assistant.use_llm else "extractive, offline"
    print(f"Mode: {mode} - type a question, or an empty line to quit.")
    while question := input("\n> ").strip():
        show(question)
