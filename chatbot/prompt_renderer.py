"""
POML Prompt Renderer

This file handles using parameters passed to it in conjunction with the POML prompt files 
to build prompts to be sent to the LLMs.
"""

from pathlib import Path
from poml import poml

ROUTING_POML = Path(__file__).parent / "prompts" / "route.poml"
ANSWER_POML = Path(__file__).parent / "prompts" / "answer.poml"
JUDGE_POML = Path(__file__).parent / "prompts" / "judge.poml"

# Typographic characters that LLM output (or pasted user input) can contain
# but that break poml's non-UTF-8 file I/O on Windows.
_UNSAFE_CHAR_REPLACEMENTS = {
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',  # curly double quotes
    "–": "-", "—": "-",  # en dash, em dash
    "…": "...",               # ellipsis
    " ": " ",                 # non-breaking space
}


def _sanitize(text: str) -> str:
    """Replace characters known to break poml's file I/O on Windows."""
    for old, new in _UNSAFE_CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text

ANSWER_TECHNIQUES = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "cot": "Chain of Thought",
    "advanced": "Advanced (Self-Consistency)"
}


def render_routing_prompt(question: str) -> str:
    poml_rendered = poml(
        str(ROUTING_POML),
        chat=True,
        format="langchain",
        # TODO: fill in the context that should be passed to the router prompt
        context={"question": _sanitize(question)}
    )

    return poml_rendered["messages"][0]["data"]["content"]


def render_answer_prompt(technique: str, question: str, 
                         context: list, language: str,
                         conversation_context: str = "") -> str:
    if technique not in ANSWER_TECHNIQUES:
        raise ValueError(
            f"Invalid technique: {technique}. Must be one of {list(ANSWER_TECHNIQUES.keys())}"
        )

    poml_rendered = poml(
        str(ANSWER_POML),
        chat=True,
        format="langchain",
        # TODO: fill in the context that should be passed to the answer prompt
        context={
            "technique": technique,
            "question": _sanitize(question),
            "context": [{**chunk, "content": _sanitize(chunk["content"])} for chunk in context],
            "language": language,
            "conversation_context": _sanitize(conversation_context),
        }
    )

    # TODO: correct the return statement to return just the content of the rendered poml prompt.
    return poml_rendered["messages"][0]["data"]["content"]


def render_judge_prompt(question: str, responses: list, comparison_type: str) -> str:
    # Build a list of dicts for each LLM response with it's label and response content
    responses_list = []
    for idx, item in enumerate(responses, 1):
        label = item.get("model_name") or item.get("technique_name") or f"Item {idx}"
        responses_list.append(
            {
                "label": _sanitize(label),
                "response": _sanitize(item.get("response", "") or ""),
            }
        )

    poml_rendered = poml(
        str(JUDGE_POML),
        chat=True,
        format="langchain",
        context={
            "question": _sanitize(question),
            "comparison_type": comparison_type,
            "responses_list": responses_list,
        }
    )

    return poml_rendered["messages"][0]["data"]["content"]


def get_answer_techniques() -> dict:
    """Returns a dictionary of the different answer prompting techniques"""
    return ANSWER_TECHNIQUES