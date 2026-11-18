"""Prompt formats shared by teacher generation, fine-tuning and inference.

Keeping one module prevents the classic bug where training prompts and runtime
prompts are formatted differently.
"""

from __future__ import annotations

import re
from typing import Sequence

PERSONALITY_INSTRUCTIONS = {
    "Friendly": "Be warm and encouraging. Use short sentences and one practical next step.",
    "Formal": "Be precise and academic. Define terms and cite the supplied source.",
    "Concise": "Be brief. Give the answer first, then one next step.",
}

CATEGORIES = ("Weak", "Developing", "Strong", "Unknown")


def format_source(source: dict) -> str:
    reference = source.get("source_ref") or source.get("page_ref") or source.get("original_name", "source")
    name = source.get("original_name") or source.get("file") or "source"
    text = re.sub(r"\s+", " ", str(source.get("text", ""))).strip()[:1400]
    return f"[{name} | {reference}] {text}"


def build_revision_prompt(question: str, sources: Sequence[dict],
                          personality: str = "Friendly") -> str:
    instruction = PERSONALITY_INSTRUCTIONS.get(personality, PERSONALITY_INSTRUCTIONS["Friendly"])
    source_block = "\n".join(format_source(source) for source in sources[:4]) or "[no source supplied]"
    return (
        f"<user> You are Rewiseorist, a local revision tutor. {instruction}\n"
        "Use only the source extracts below. If they do not contain the answer, say that clearly.\n"
        "Answer the question in 3 to 5 short sentences. End with one citation line in the form\n"
        "Source: <file> | <page/section>.\n"
        f"Question: {question.strip()}\n"
        f"<source>\n{source_block}\n"
        "<assistant>"
    )


def build_feedback_prompt(question: str, source: dict, student_answer: str,
                          personality: str = "Friendly") -> str:
    instruction = PERSONALITY_INSTRUCTIONS.get(personality, PERSONALITY_INSTRUCTIONS["Friendly"])
    return (
        f"<user> You are Rewiseorist, a local revision tutor. {instruction}\n"
        "Mark the student's answer against the source. Say what is correct, correct one mistake,\n"
        "and give one next step. Do not invent facts. End with a citation line.\n"
        f"Question: {question.strip()}\n"
        f"Student answer: {student_answer.strip()}\n"
        f"<source>\n{format_source(source)}\n"
        "<assistant>"
    )


def build_quiz_prompt(topic: str, source: dict, difficulty: str = "medium") -> str:
    return (
        "<user> You are an exam writer. Create one checked short-answer question from the source.\n"
        "Return JSON only with keys: question, answer, hint, explanation, source_ref.\n"
        f"Topic: {topic}. Difficulty: {difficulty}.\n"
        f"<source>\n{format_source(source)}\n"
        "<assistant>"
    )


def build_performance_prompt(features: dict, history_summary: str) -> str:
    return (
        "<user> You are a student-performance assessor. Classify this revision history as "
        "Weak, Developing, Strong or Unknown.\n"
        "Rules: use Unknown when there are fewer than three distinct checked questions. "
        "Consider accuracy, trend, hints, independent answers, difficulty and repeated errors. "
        "Never label a student permanently from one quiz.\n"
        "Return JSON only: {\"category\":\"Weak|Developing|Strong|Unknown\","
        "\"reason\":\"one or two evidence-based sentences\"}.\n"
        f"Features: {features}\n"
        f"Attempt history: {history_summary}\n"
        "<assistant>"
    )


def build_ollama_revision_messages(question: str, sources: Sequence[dict],
                                   personality: str = "Friendly",
                                   max_sources: int = 3,
                                   max_chars: int = 600) -> list[dict]:
    """Short chat messages for the Ollama SmolLM2 revision assistant.

    SmolLM2 360M has a 2048-token context, so only the most relevant
    retrieved passages are included and each is capped in length.
    """
    instruction = PERSONALITY_INSTRUCTIONS.get(personality, PERSONALITY_INSTRUCTIONS["Friendly"])
    system = (
        "You are Rewiseorist, a local revision tutor. " + instruction +
        " Answer only from the numbered sources. If they do not contain the answer, "
        "say clearly that the uploaded material does not contain it. "
        "Write 2 to 3 short sentences and end with one line: "
        "Source: <file> | <section>."
    )
    parts: list[str] = []
    for index, source in enumerate(sources[:max_sources], 1):
        name = source.get("original_name") or source.get("file") or "source"
        ref = source.get("source_ref") or source.get("page_ref") or ""
        text = re.sub(r"\s+", " ", str(source.get("text", ""))).strip()[:max_chars]
        parts.append(f"[{index}] {name} | {ref}\n{text}")
    source_block = "\n".join(parts) or "No uploaded source is available."
    user = f"Question: {question.strip()}\n\nSources:\n{source_block}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


CHAT_MARKER_RE = re.compile(r"^<user>\s*|\s*<assistant>\s*$", re.IGNORECASE)


def strip_chat_markers(prompt: str) -> str:
    """Return the instruction body for an instruct teacher's chat endpoint.

    The tiny student model is trained with explicit `<user>`/`<assistant>`
    boundaries. Instruct teacher models apply their own chat template, so those
    markers should not be sent twice.
    """
    text = (prompt or "").strip()
    text = re.sub(r"^<user>\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*<assistant>\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


CITATION_RE = re.compile(r"source\s*:\s*([^\n|]+)\|\s*([^\n]+)", re.IGNORECASE)


def extract_citation(text: str) -> str:
    match = CITATION_RE.search(text or "")
    if not match:
        return ""
    return f"{match.group(1).strip()} | {match.group(2).strip()}"


def clean_teacher_answer(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Teacher chat templates sometimes leak a stop token; remove it.
    text = text.replace("<|eot_id|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()
