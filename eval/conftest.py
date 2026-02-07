"""
Pytest fixtures and configuration for eval tests.
Usage: pytest eval/ -v
"""

import pytest

from eval.datasets import TEST_CASES


@pytest.fixture
def sample_ticker():
    """Return first test ticker."""
    return TEST_CASES[0]["ticker"]


@pytest.fixture
def all_tickers():
    """Return all test tickers."""
    return [case["ticker"] for case in TEST_CASES]


@pytest.fixture
def expected_tools():
    """Return expected tool list."""
    return ["get_company_profile", "get_financial_ratios", "get_financial_statements"]


@pytest.fixture
def expected_sections():
    """Return expected report sections."""
    return ["Company Overview", "Financial Health", "Recommendation"]
