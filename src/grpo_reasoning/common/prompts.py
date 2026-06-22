from __future__ import annotations

import re

R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""


def build_chat_prompt(
    question: str,
    task_instructions: str,
    few_shot_question: str | None = None,
    few_shot_answer: str | None = None,
    few_shot_examples: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Build a chat-format prompt list.

    Args:
        question: User question to place at the end of the prompt.
        task_instructions: Optional task-specific system instructions.
        few_shot_question: Optional single example user question (legacy single-shot).
        few_shot_answer: Optional single example assistant answer (legacy single-shot).
        few_shot_examples: Optional ordered list of (question, answer) pairs for
            multi-shot prompting. Takes precedence over the single-shot pair.

    Returns:
        Chat messages in tokenizer-compatible dictionary format.
    """
    system_content = R1_STYLE_SYSTEM_PROMPT
    if task_instructions:
        system_content = f"{system_content}\n{task_instructions}"

    messages: list[dict] = [{"role": "system", "content": system_content}]

    examples = list(few_shot_examples) if few_shot_examples else []
    if not examples and few_shot_question is not None and few_shot_answer is not None:
        examples = [(few_shot_question, few_shot_answer)]
    for example_question, example_answer in examples:
        messages.append({"role": "user", "content": example_question})
        messages.append({"role": "assistant", "content": example_answer})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def extract_xml_answer(text: str) -> str:
    """Extract answer text from XML-style answer tags.

    Args:
        text: Completion text that may contain `<answer>...</answer>`.

    Returns:
        Extracted answer content, with surrounding Markdown fences removed when present.
    """
    try:
        content = text.split("<answer>")[-1].split("</answer>")[0].strip()
    except (IndexError, AttributeError):
        return ""
    content = re.sub(r"^```[a-zA-Z]*\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()
