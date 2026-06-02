"""The single prompt template used for training *and* evaluation.

Keeping one definition guarantees the model is scored with the exact format it
was trained on. We use the classic Alpaca-style instruction template.
"""

from __future__ import annotations

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def build_prompt(instruction: str, context: str | None) -> str:
    """Render the prompt *without* the answer (used at inference/eval time)."""
    context = (context or "").strip()
    if context:
        return PROMPT_WITH_INPUT.format(instruction=instruction.strip(), input=context)
    return PROMPT_NO_INPUT.format(instruction=instruction.strip())


def build_example_text(instruction: str, context: str | None, response: str) -> str:
    """Render the full training text: prompt + answer + EOS marker.

    The trailing newline keeps a clean boundary; the tokenizer adds EOS.
    """
    return build_prompt(instruction, context) + response.strip() + "\n"
