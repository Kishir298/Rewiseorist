from ml.prompts import (
    build_performance_prompt,
    build_revision_prompt,
    clean_teacher_answer,
    extract_citation,
)
from ml.teacher_client import TEACHER_SPECS, TeacherSpec, build_command


def test_revision_prompt_includes_source_and_personality():
    prompt = build_revision_prompt(
        "What is friction?",
        [{"original_name": "forces.md", "source_ref": "forces.md|p2",
          "text": "Friction opposes motion."}],
        "Formal",
    )
    assert "<user>" in prompt
    assert "forces.md" in prompt
    assert "Friction opposes motion." in prompt
    assert "Formal" not in prompt  # the instruction text is rendered, not the label
    assert "Source:" in prompt


def test_feedback_and_performance_prompts():
    source = {"original_name": "forces.md", "page_ref": "2", "text": "Friction opposes motion."}
    from ml.prompts import build_feedback_prompt
    feedback = build_feedback_prompt("What is friction?", source, "Friction helps motion.")
    assert "Student answer" in feedback
    performance = build_performance_prompt({"mean_accuracy": 0.5}, "attempt 1: 0/1")
    assert "Return JSON only" in performance
    assert "Unknown" in performance


def test_citation_and_cleanup():
    assert extract_citation("Answer.\nSource: forces.md | page 2") == "forces.md | page 2"
    assert extract_citation("No citation here") == ""
    assert clean_teacher_answer("```text\nHello\n```") == "Hello"
    assert "<|eot_id|>" not in clean_teacher_answer("Hello <|eot_id|>")


def test_server_command_is_independent_runtime(tmp_path):
    spec = TeacherSpec(
        key="test", name="Test", filename="test.gguf", licence="test",
        source_url="https://example.invalid",
    )
    command = build_command(spec, port=8099, context=2048, threads=4)
    joined = " ".join(command).lower()
    assert "llama-server" in joined
    assert "-m" in command
    assert "8099" in command
    assert "teacher_models" in joined
    assert len(TEACHER_SPECS) >= 2
