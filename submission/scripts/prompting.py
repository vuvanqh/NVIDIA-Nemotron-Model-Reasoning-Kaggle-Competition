"""Prompt utilities for local Nemotron reasoning experiments."""

from __future__ import annotations


def build_reasoning_prompt(puzzle: str) -> str:
    """Build a compact reasoning prompt that requires one final boxed answer."""
    cleaned_puzzle = str(puzzle).strip()
    return (
        "Solve the puzzle. Give concise reasoning, then provide the final answer.\n"
        "The final answer must appear in exactly one \\boxed{...} expression.\n"
        "Do not put intermediate values in boxes.\n\n"
        f"Puzzle:\n{cleaned_puzzle}\n\n"
        "Answer:"
    )
