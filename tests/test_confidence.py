from __future__ import annotations

import math

import pytest

from snuaichal.confidence import align_answer_confidence


class PieceTokenizer:
    def __init__(self, merged: tuple[str, ...] = ()) -> None:
        pieces = sorted(set(merged), key=len, reverse=True)
        characters = list("[]1234, abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:\n")
        self.pieces = pieces + [character for character in characters if character not in pieces]
        self.piece_to_id = {piece: index + 1 for index, piece in enumerate(self.pieces)}
        self.id_to_piece = {value: key for key, value in self.piece_to_id.items()}
        self.eos_token_id = 999
        self.all_special_ids = [self.eos_token_id]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result: list[int] = []
        position = 0
        while position < len(text):
            piece = next(
                (candidate for candidate in self.pieces if text.startswith(candidate, position)),
                None,
            )
            if piece is None:
                raise ValueError(f"unsupported test character: {text[position]!r}")
            result.append(self.piece_to_id[piece])
            position += len(piece)
        return result

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        return "".join(
            self.id_to_piece[token_id]
            for token_id in token_ids
            if not (skip_special_tokens and token_id in self.all_special_ids)
        )


def align(text: str, tokenizer: PieceTokenizer, raw_output: str | None = None):
    token_ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
    logprobs = [-(index + 1) / 10 for index in range(len(token_ids))]
    return align_answer_confidence(
        raw_output=text if raw_output is None else raw_output,
        generated_token_ids=token_ids,
        generated_token_logprobs=logprobs,
        tokenizer=tokenizer,
    )


@pytest.mark.parametrize(
    "text",
    [
        "[1, 2, 3, 4]",
        "Answer: [1, 2, 3, 4]",
        "[1, 2, 3, 4] done",
        "\n [1, 2, 3, 4]",
    ],
)
def test_answer_alignment_handles_plain_prose_and_leading_whitespace(text: str) -> None:
    tokenizer = PieceTokenizer()

    result = align(text, tokenizer)

    assert result.answer_confidence_valid is True
    assert result.parsed_answer_substring == "[1, 2, 3, 4]"
    assert result.answer_token_ids == tokenizer.encode("[1, 2, 3, 4]")
    assert len(result.answer_token_ids) == len(result.answer_token_logprobs)
    assert math.isclose(result.answer_logprob_sum, sum(result.answer_token_logprobs))
    assert math.isclose(
        result.answer_logprob_mean,
        sum(result.answer_token_logprobs) / len(result.answer_token_logprobs),
    )
    assert result.alignment_failure_reason is None


def test_answer_alignment_uses_last_valid_bracketed_permutation() -> None:
    tokenizer = PieceTokenizer()

    result = align("first [1, 2, 3, 4] then [4, 3, 2, 1]", tokenizer)

    assert result.parsed_answer_substring == "[4, 3, 2, 1]"
    assert result.answer_token_ids == tokenizer.encode("[4, 3, 2, 1]")


def test_answer_alignment_supports_tokens_that_merge_brackets_and_numbers() -> None:
    tokenizer = PieceTokenizer(("[1", ", 2", ", 3", ", 4", "]"))

    result = align("Answer: [1, 2, 3, 4]", tokenizer)

    assert result.answer_confidence_valid is True
    assert result.answer_token_ids == tokenizer.encode("[1, 2, 3, 4]")
    assert result.alignment_method in {"exact_subsequence", "incremental_decode"}


def test_answer_alignment_ignores_special_tokens() -> None:
    tokenizer = PieceTokenizer()
    answer_ids = tokenizer.encode("[1, 2, 3, 4]")
    token_ids = [tokenizer.eos_token_id, *answer_ids, tokenizer.eos_token_id]
    logprobs = [-9.0, *[-0.1] * len(answer_ids), -8.0]

    result = align_answer_confidence(
        raw_output="[1, 2, 3, 4]",
        generated_token_ids=token_ids,
        generated_token_logprobs=logprobs,
        tokenizer=tokenizer,
    )

    assert result.answer_token_ids == answer_ids
    assert result.answer_token_logprobs == [-0.1] * len(answer_ids)


def test_answer_alignment_marks_decode_mismatch_invalid() -> None:
    tokenizer = PieceTokenizer()
    text = "[1, 2, 3, 4]"
    token_ids = tokenizer.encode(text)

    result = align_answer_confidence(
        raw_output="prefix [1, 2, 3, 4]",
        generated_token_ids=token_ids,
        generated_token_logprobs=[-0.1] * len(token_ids),
        tokenizer=tokenizer,
    )

    assert result.answer_confidence_valid is False
    assert result.answer_logprob_mean is None
    assert result.alignment_failure_reason is not None


@pytest.mark.parametrize("text", ["not a list", "[1, 2, 3]", "[1, 1, 3, 4]"])
def test_answer_alignment_marks_malformed_output_invalid(text: str) -> None:
    tokenizer = PieceTokenizer()

    result = align(text, tokenizer)

    assert result.answer_confidence_valid is False
    assert result.parsed_answer_substring is None
    assert result.answer_token_ids == []
    assert result.alignment_failure_reason == "no valid permutation substring"
