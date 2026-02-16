"""
Dataset management for LangSmith evaluations.
Handles test case definitions and dataset creation.
"""

from langsmith import Client

# Current active tools from src/graph.py
EXPECTED_TOOLS = [
    "get_company_profile",
    "get_financial_ratios",
    "get_financial_statements",
    "get_earnings_summary_via_search",
    "get_revenue_segmentation",
    "get_analyst_estimates",
    "get_ownership_via_search",
]

# Report sections matching writer prompt in src/graph.py
EXPECTED_SECTIONS = [
    "Executive Summary",
    "Business Transformation",
    "The Moat",
    "Financial Performance",
    "Outlook",
    "Bear Case",
    "Valuation",
]

# Test cases with expected behaviors
TEST_CASES: list[dict] = [
    {
        "ticker": "AAPL",
        "expected_tools": EXPECTED_TOOLS,
        "expected_sections": EXPECTED_SECTIONS,
    },
    {
        "ticker": "NVDA",
        "expected_tools": EXPECTED_TOOLS,
        "expected_sections": EXPECTED_SECTIONS,
    },
    {
        "ticker": "MSFT",
        "expected_tools": EXPECTED_TOOLS,
        "expected_sections": EXPECTED_SECTIONS,
    },
    {
        "ticker": "GOOGL",
        "expected_tools": EXPECTED_TOOLS,
        "expected_sections": EXPECTED_SECTIONS,
    },
    {
        "ticker": "TSLA",
        "expected_tools": EXPECTED_TOOLS,
        "expected_sections": EXPECTED_SECTIONS,
    },
]


def _create_dataset(
    client: Client,
    dataset_name: str,
    description: str,
) -> str:
    """Create a new dataset and populate with TEST_CASES. Returns dataset name."""
    dataset = client.create_dataset(dataset_name=dataset_name, description=description)
    print(f"Created new dataset: {dataset_name}")

    for case in TEST_CASES:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"ticker": case["ticker"]},
            outputs={
                "expected_tools": case["expected_tools"],
                "expected_sections": case["expected_sections"],
            },
        )
    print(f"Added {len(TEST_CASES)} test cases")
    return dataset.name


def get_or_create_dataset(
    dataset_name: str = "financial-agent-evals",
    description: str = "Eval dataset for financial research agent",
) -> str:
    """Get existing dataset or create new one. Returns dataset name."""
    client = Client()

    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
        return dataset.name
    except Exception:
        pass

    return _create_dataset(client, dataset_name, description)


def recreate_dataset(
    dataset_name: str = "financial-agent-evals",
    description: str = "Eval dataset for financial research agent",
) -> str:
    """Delete existing dataset and recreate with current expected values."""
    client = Client()

    try:
        client.delete_dataset(dataset_name=dataset_name)
        print(f"Deleted old dataset: {dataset_name}")
    except Exception:
        print(f"No existing dataset to delete: {dataset_name}")

    return _create_dataset(client, dataset_name, description)


def add_custom_ticker(dataset_name: str, ticker: str) -> None:
    """Add custom ticker to existing dataset."""
    client = Client()
    dataset = client.read_dataset(dataset_name=dataset_name)

    client.create_example(
        dataset_id=dataset.id,
        inputs={"ticker": ticker},
        outputs={
            "expected_tools": EXPECTED_TOOLS,
            "expected_sections": EXPECTED_SECTIONS,
        },
    )
    print(f"Added ticker {ticker} to dataset")
