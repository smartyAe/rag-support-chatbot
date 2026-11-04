from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docs_assistant.answer import NO_ANSWER, Assistant
from docs_assistant.ingest import chunk_markdown
from docs_assistant.retrieve import stem, tokenize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def assistant():
    return Assistant(ROOT / "docs_kb", use_llm=False)


def test_chunks_keep_title_and_section():
    chunks = chunk_markdown("# Returns\n\n## Return period\nThirty days.\n\n## Exchanges\nNo.", "r.md")
    assert [(c.title, c.section, c.text) for c in chunks] == [
        ("Returns", "Return period", "Thirty days."), ("Returns", "Exchanges", "No.")]
    assert chunks[0].citation == "Returns › Return period"


@pytest.mark.parametrize("a, b", [("invoices", "invoice"), ("returned", "return"),
                                  ("shipping", "ship"), ("delivered", "delivery")])
def test_stemming_groups_word_forms(a, b):
    assert stem(a) == stem(b)


def test_stopwords_and_accents_are_removed():
    assert tokenize("What is the Délivery time?") == [stem("delivery"), "tim"]


def test_answer_quotes_the_documentation_with_its_source(assistant):
    answer = assistant.ask("How long does delivery take to Abidjan?")
    assert answer.answered
    assert "5 to 8 business days" in answer.text
    assert answer.sources[0].section == "Delivery times"


def test_customer_wording_is_matched_through_the_glossary(assistant):
    answer = assistant.ask("My package arrived broken, what should I do?")
    assert answer.sources[0].section == "Damaged or missing parcels"


def test_out_of_scope_question_is_refused(assistant):
    answer = assistant.ask("Who is the CEO of the company?")
    assert not answer.answered
    assert answer.text == NO_ANSWER and answer.sources == []


def test_retrieval_quality_does_not_regress():
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from evaluate import evaluate

    result = evaluate(verbose=False)
    assert result["hit@3"] >= 0.85
    assert result["out_of_scope_refused"] == "5/5"


def test_api_returns_answer_and_sources():
    from docs_assistant import app as web

    web.assistant = Assistant(ROOT / "docs_kb", use_llm=False)
    client = TestClient(web.app)
    body = client.post("/api/ask", json={"question": "When is an invoice due?"}).json()
    assert body["answered"] and "30 days" in body["answer"]
    assert body["sources"][0]["section"] == "Paying by invoice"
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    page = client.get("/", params={"q": ["Is shipping free?"]}).text
    assert "150 EUR" in page


class _FakeOpenAI:
    """Stands in for the OpenAI client: no network call, no API key."""

    last_messages: list = []

    def __init__(self, reply: str):
        self.reply = reply
        self.chat = self
        self.completions = self

    def create(self, model, temperature, messages):
        _FakeOpenAI.last_messages = messages
        message = type("M", (), {"content": self.reply})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


def test_llm_mode_sends_sources_and_keeps_cited_ones(monkeypatch):
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda: _FakeOpenAI("Free above 150 EUR excluding tax [1]."))
    answer = Assistant(ROOT / "docs_kb", use_llm=True).ask("Is shipping free?")
    prompt = _FakeOpenAI.last_messages[1]["content"]
    assert "[1] Shipping and delivery › Shipping costs" in prompt
    assert answer.mode == "llm" and answer.sources[0].section == "Shipping costs"


def test_llm_mode_passes_through_a_refusal(monkeypatch):
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda: _FakeOpenAI(NO_ANSWER))
    answer = Assistant(ROOT / "docs_kb", use_llm=True).ask("Can I return an opened toner cartridge?")
    assert not answer.answered and answer.sources == []


def test_api_key_in_environment_does_not_enable_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    monkeypatch.delenv("ASSISTANT_MODE", raising=False)
    assert Assistant(ROOT / "docs_kb").use_llm is False


def test_llm_failure_falls_back_to_the_documents(monkeypatch):
    import openai

    def broken():
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(openai, "OpenAI", broken)
    answer = Assistant(ROOT / "docs_kb", use_llm=True).ask("When is an invoice due?")
    assert answer.answered and answer.mode == "extractive" and "30 days" in answer.text
