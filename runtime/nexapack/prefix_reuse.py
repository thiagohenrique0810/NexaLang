"""Pure page-prefix arithmetic for KV shared between unrelated sequences.

A complete KV page holds the state of the tokens at its own absolute positions
and of nothing else: not of how a sequence reached them, and not of kinship.
Two sessions may therefore share a page whenever their prompts agree on every
token that page covers. This module answers exactly that question, and answers
it without a session, an allocation or a file, so the rule can be tested on its
own instead of through an executor.
"""
from __future__ import annotations


def _token_sequence(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple of token IDs")
    for token in value:
        # bool is an int subclass; a flag that reached a token list is a bug,
        # not a token, so identity of the type is what is checked.
        if type(token) is not int:
            raise ValueError(f"{label} must contain only integer token IDs")
    return value


def common_prefix_length(tokens_a, tokens_b):
    """How many leading tokens the two sequences agree on."""
    length = 0
    for left, right in zip(_token_sequence(tokens_a, "tokens_a"),
                           _token_sequence(tokens_b, "tokens_b")):
        if left != right:
            break
        length += 1
    return length


def common_page_prefix(tokens_a, tokens_b, page_tokens):
    """Complete pages both sequences would fill with identical tokens.

    The trailing partial page is deliberately excluded: its owner keeps writing
    into it, so only a whole page is immutable and therefore shareable. The
    count is the floor of the common token prefix over the page size.
    """
    if type(page_tokens) is not int or page_tokens <= 0:
        raise ValueError("page_tokens must be a positive integer")
    return common_prefix_length(tokens_a, tokens_b) // page_tokens
