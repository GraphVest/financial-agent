"""
Pytest tests for eval functionality.
Usage: pytest eval/test_eval.py -v
"""

import pytest

from eval.evaluators import _get_final_output, completeness_evaluator
from eval.runner import run_single_eval


class TestCompletenessEvaluator:
    """Test completeness evaluator logic."""

    def test_all_sections_present(self):
        """Should score 1.0 when all sections present."""
        # Mock run with all sections
        class MockRun:
            outputs = {
                "output": """
                # AAPL Investment Report
                ## Company Overview
                Apple Inc is a tech company.
                ## Financial Health
                Revenue is growing.
                ## Recommendation
                Buy - strong fundamentals.
                """
            }

        class MockExample:
            outputs = {
                "expected_sections": ["Company Overview", "Financial Health", "Recommendation"]
            }

        result = completeness_evaluator(MockRun(), MockExample())
        assert result.score == 1.0

    def test_missing_section(self):
        """Should score < 1.0 when sections missing."""

        class MockRun:
            outputs = {
                "output": """
                # Report
                ## Company Overview
                Some info.
                ## Recommendation
                Buy.
                """
            }

        class MockExample:
            outputs = {
                "expected_sections": ["Company Overview", "Financial Health", "Recommendation"]
            }

        result = completeness_evaluator(MockRun(), MockExample())
        assert result.score < 1.0
        assert "Financial Health" in result.comment


class TestHelpers:
    """Test helper functions."""

    def test_get_final_output_string(self):
        """Should extract string output."""

        class MockRun:
            outputs = {"output": "Hello world"}

        assert _get_final_output(MockRun()) == "Hello world"

    def test_get_final_output_empty(self):
        """Should return empty for no outputs."""

        class MockRun:
            outputs = None

        assert _get_final_output(MockRun()) == ""


@pytest.mark.skip(reason="Requires API calls - run manually")
class TestIntegration:
    """Integration tests requiring API calls."""

    def test_single_eval_aapl(self):
        """Run single eval on AAPL."""
        results = run_single_eval("AAPL")
        assert "faithfulness" in results
        assert "completeness" in results
        assert "tool_coverage" in results
