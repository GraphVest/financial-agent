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
        # Mock run with all sections matching writer prompt
        class MockRun:
            outputs = {
                "output": """
                # AAPL Deep Dive: The Services Pivot

                **🚨 Executive Summary (TL;DR)**
                Apple is transforming.

                **1. Business Transformation: From Hardware to Services**
                Revenue mix is shifting.

                **2. The Moat & Competitive Advantage: Ecosystem Lock-in**
                iPhone ecosystem is unmatched.

                **3. Financial Performance: Margins Expanding**
                Operating leverage is real.

                **4. Outlook & Future Roadmap: Vision Pro Era**
                Management guided higher.

                **5. The Bear Case & Risks: China Dependency**
                Geopolitical risks remain.

                **6. Valuation & The Verdict: Priced for Perfection**
                Forward P/E suggests premium.
                """
            }

        class MockExample:
            outputs = {
                "expected_sections": [
                    "Executive Summary",
                    "Business Transformation",
                    "The Moat",
                    "Financial Performance",
                    "Outlook",
                    "Bear Case",
                    "Valuation",
                ]
            }

        result = completeness_evaluator(MockRun(), MockExample())
        assert result.score == 1.0

    def test_missing_section(self):
        """Should score < 1.0 when sections missing."""

        class MockRun:
            outputs = {
                "output": """
                # Report
                **🚨 Executive Summary (TL;DR)**
                Some info.
                **6. Valuation & The Verdict**
                Buy.
                """
            }

        class MockExample:
            outputs = {
                "expected_sections": [
                    "Executive Summary",
                    "Business Transformation",
                    "The Moat",
                    "Financial Performance",
                    "Outlook",
                    "Bear Case",
                    "Valuation",
                ]
            }

        result = completeness_evaluator(MockRun(), MockExample())
        assert result.score < 1.0
        assert "Business Transformation" in result.comment


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


@pytest.mark.skip(reason="Requires Langfuse API - run manually")
class TestLangfuseIntegration:
    """Langfuse integration tests requiring API calls."""

    def test_langfuse_single_eval(self):
        """Run Langfuse eval on single ticker."""
        from eval.langfuse_runner import run_langfuse_eval

        results = run_langfuse_eval("AAPL")
        assert "faithfulness" in results
        assert "completeness" in results
        assert "tool_coverage" in results
