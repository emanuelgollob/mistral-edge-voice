"""
Unit tests for the speculation snapshot-match path in voice_agent.

Covers:
  - _normalize_for_match: pure normalization (lower / strip / collapse /
    drop trailing punctuation).
  - snapshot-match decision: whether two strings are treated as the same
    turn under match normalization. The decision in voice_agent.run() is
    exactly `_normalize_for_match(snap) == _normalize_for_match(final)`,
    so we mirror that here as `_snapshots_match`.

Run with:
    python -m unittest discover tests/
"""

import unittest

from voice_agent import _normalize_for_match


def _snapshots_match(snap: str, final: str) -> bool:
    """Mirror of the inline hit/miss decision in voice_agent.run()."""
    return _normalize_for_match(snap) == _normalize_for_match(final)


class NormalizeForMatchTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(_normalize_for_match("hello"), "hello")

    def test_lowercases(self):
        self.assertEqual(_normalize_for_match("Hello"), "hello")
        self.assertEqual(_normalize_for_match("HELLO"), "hello")

    def test_strips_leading_and_trailing_whitespace(self):
        self.assertEqual(_normalize_for_match("  hello  "), "hello")

    def test_collapses_internal_whitespace(self):
        self.assertEqual(_normalize_for_match("hello   world"), "hello world")

    def test_normalizes_newlines_and_tabs_as_whitespace(self):
        self.assertEqual(_normalize_for_match("hello\nworld"), "hello world")
        self.assertEqual(_normalize_for_match("hello\tworld"), "hello world")

    def test_strips_single_trailing_punctuation(self):
        self.assertEqual(_normalize_for_match("hello."), "hello")
        self.assertEqual(_normalize_for_match("hello?"), "hello")
        self.assertEqual(_normalize_for_match("hello!"), "hello")

    def test_strips_run_of_trailing_punctuation(self):
        self.assertEqual(_normalize_for_match("hello!?"), "hello")
        self.assertEqual(_normalize_for_match("hello..."), "hello")

    def test_preserves_internal_punctuation(self):
        self.assertEqual(_normalize_for_match("don't go"), "don't go")
        self.assertEqual(_normalize_for_match("hello, world."), "hello, world")

    def test_empty_string(self):
        self.assertEqual(_normalize_for_match(""), "")

    def test_whitespace_only(self):
        self.assertEqual(_normalize_for_match("   "), "")


class SnapshotMatchDecisionTests(unittest.TestCase):
    """Speculation hit / miss decision: snap matches final under
    normalization?

    Hits should fire when Voxtral's transcription.delta accumulation
    differs only cosmetically from the final transcription.done text
    (case, whitespace, trailing period). Misses should fire when the
    user actually said something different from the snapshot.
    """

    def test_identical_strings_match(self):
        self.assertTrue(_snapshots_match("Hello world", "Hello world"))

    def test_trailing_period_matches(self):
        # The canonical Voxtral case: delta accumulates without trailing
        # punctuation, the done event adds a period.
        self.assertTrue(_snapshots_match("Hello world", "Hello world."))

    def test_case_differences_match(self):
        self.assertTrue(_snapshots_match("hello world", "Hello World"))

    def test_trailing_whitespace_matches(self):
        self.assertTrue(_snapshots_match("Hello world", "Hello world  "))

    def test_extra_internal_space_matches(self):
        self.assertTrue(_snapshots_match("Hello  world", "Hello world"))

    def test_combined_cosmetic_differences_match(self):
        self.assertTrue(_snapshots_match("  hello world  ", "Hello world."))

    def test_user_added_word_is_miss(self):
        # Speculation fired on a prefix; user kept talking. Must miss
        # so the agent doesn't reply to half a question.
        self.assertFalse(_snapshots_match("What time", "What time is it"))

    def test_typo_is_miss(self):
        self.assertFalse(_snapshots_match("hello world", "hella world"))

    def test_internal_punctuation_difference_is_miss(self):
        # Trailing punctuation is stripped; internal punctuation isn't.
        # A missing comma changes meaning enough to discard the spec.
        self.assertFalse(_snapshots_match("Hello, world", "Hello world"))


if __name__ == "__main__":
    unittest.main()
