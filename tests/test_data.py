from common.prompt import build_example_text, build_prompt


def test_prompt_with_and_without_input():
    with_ctx = build_prompt("Summarise.", "Long context here.")
    assert "### Input:" in with_ctx
    assert with_ctx.rstrip().endswith("### Response:")

    no_ctx = build_prompt("Say hi.", "")
    assert "### Input:" not in no_ctx
    assert no_ctx.rstrip().endswith("### Response:")


def test_example_text_contains_response():
    text = build_example_text("Q?", None, "A.")
    assert "### Response:\nA." in text


def test_sample_jsonl_matches_schema():
    import json
    from pathlib import Path

    lines = Path("data/sample.jsonl").read_text().splitlines()
    assert lines, "sample.jsonl should not be empty"
    for line in lines:
        row = json.loads(line)
        assert {"instruction", "input", "output", "text"} <= set(row)
        assert row["text"].endswith("\n")
