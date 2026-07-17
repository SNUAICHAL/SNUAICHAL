"""Tokenizer-level alignment of generated confidence to the final valid answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from snuaichal.submission import find_last_valid_permutation_substring


@dataclass(frozen=True)
class AnswerConfidence:
    parsed_answer_substring: str | None
    answer_token_start: int | None
    answer_token_end: int | None
    answer_token_ids: list[int]
    answer_token_logprobs: list[float]
    answer_logprob_sum: float | None
    answer_logprob_mean: float | None
    answer_confidence_valid: bool
    alignment_method: str | None
    alignment_failure_reason: str | None


def _invalid(substring: str | None, reason: str) -> AnswerConfidence:
    return AnswerConfidence(
        parsed_answer_substring=substring,
        answer_token_start=None,
        answer_token_end=None,
        answer_token_ids=[],
        answer_token_logprobs=[],
        answer_logprob_sum=None,
        answer_logprob_mean=None,
        answer_confidence_valid=False,
        alignment_method=None,
        alignment_failure_reason=reason,
    )


def _find_last_subsequence(values: Sequence[int], target: Sequence[int]) -> int | None:
    if not target or len(target) > len(values):
        return None
    for start in range(len(values) - len(target), -1, -1):
        if list(values[start : start + len(target)]) == list(target):
            return start
    return None


def _success(
    *,
    substring: str,
    start: int,
    end: int,
    token_ids: Sequence[int],
    token_logprobs: Sequence[float],
    special_ids: set[int],
    method: str,
) -> AnswerConfidence:
    selected = [
        (token_id, float(logprob))
        for token_id, logprob in zip(
            token_ids[start:end], token_logprobs[start:end], strict=True
        )
        if token_id not in special_ids
    ]
    if not selected:
        return _invalid(substring, "aligned span contains no answer tokens")
    ids = [token_id for token_id, _ in selected]
    logprobs = [logprob for _, logprob in selected]
    total = sum(logprobs)
    return AnswerConfidence(
        parsed_answer_substring=substring,
        answer_token_start=start,
        answer_token_end=end,
        answer_token_ids=ids,
        answer_token_logprobs=logprobs,
        answer_logprob_sum=total,
        answer_logprob_mean=total / len(logprobs),
        answer_confidence_valid=True,
        alignment_method=method,
        alignment_failure_reason=None,
    )


def align_answer_confidence(
    *,
    raw_output: str,
    generated_token_ids: Sequence[int],
    generated_token_logprobs: Sequence[float],
    tokenizer: Any,
) -> AnswerConfidence:
    """Align scores to only the last valid bracketed permutation in generation."""
    located = find_last_valid_permutation_substring(raw_output)
    if located is None:
        return _invalid(None, "no valid permutation substring")
    substring, character_start, character_end = located
    if len(generated_token_ids) != len(generated_token_logprobs):
        return _invalid(substring, "generated token/logprob length mismatch")

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    try:
        decoded = tokenizer.decode(
            list(generated_token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        return _invalid(substring, f"generated token decode failed: {exc}")
    if decoded != raw_output:
        return _invalid(substring, "decoded generated tokens do not match raw output")

    try:
        candidate_ids = tokenizer.encode(substring, add_special_tokens=False)
    except Exception:
        candidate_ids = []
    exact_start = _find_last_subsequence(generated_token_ids, candidate_ids)
    if exact_start is not None:
        return _success(
            substring=substring,
            start=exact_start,
            end=exact_start + len(candidate_ids),
            token_ids=generated_token_ids,
            token_logprobs=generated_token_logprobs,
            special_ids=special_ids,
            method="exact_subsequence",
        )

    prefix_texts = [""]
    try:
        for end in range(1, len(generated_token_ids) + 1):
            prefix_texts.append(
                tokenizer.decode(
                    list(generated_token_ids[:end]),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    except Exception as exc:
        return _invalid(substring, f"incremental decode failed: {exc}")

    overlapping: list[int] = []
    for index, (before, after) in enumerate(zip(prefix_texts, prefix_texts[1:])):
        before_length = len(before)
        after_length = len(after)
        if before_length < character_end and after_length > character_start:
            outside_prefix = after[before_length:character_start]
            outside_suffix = after[character_end:after_length]
            if (outside_prefix and not outside_prefix.isspace()) or (
                outside_suffix and not outside_suffix.isspace()
            ):
                return _invalid(
                    substring,
                    "answer boundary shares a token with non-whitespace prose",
                )
            overlapping.append(index)
    if not overlapping:
        return _invalid(substring, "incremental decode could not locate answer span")
    return _success(
        substring=substring,
        start=min(overlapping),
        end=max(overlapping) + 1,
        token_ids=generated_token_ids,
        token_logprobs=generated_token_logprobs,
        special_ids=special_ids,
        method="incremental_decode",
    )
