"""
Dataset management for LangSmith evaluations.
Handles test case definitions and dataset creation.
"""

from langsmith import Client

# Test cases with expected behaviors
TEST_CASES: list[dict] = [
    {
        "ticker": "AAPL",
        "expected_tools": ["get_company_profile", "get_financial_ratios", "get_financial_statements"],
        "expected_sections": ["Company Overview", "Financial Health", "Recommendation"],
    },
    {
        "ticker": "NVDA",
        "expected_tools": ["get_company_profile", "get_financial_ratios", "get_financial_statements"],
        "expected_sections": ["Company Overview", "Financial Health", "Recommendation"],
    },
    {
        "ticker": "MSFT",
        "expected_tools": ["get_company_profile", "get_financial_ratios", "get_financial_statements"],
        "expected_sections": ["Company Overview", "Financial Health", "Recommendation"],
    },
    {
        "ticker": "GOOGL",
        "expected_tools": ["get_company_profile", "get_financial_ratios", "get_financial_statements"],
        "expected_sections": ["Company Overview", "Financial Health", "Recommendation"],
    },
    {
        "ticker": "TSLA",
        "expected_tools": ["get_company_profile", "get_financial_ratios", "get_financial_statements"],
        "expected_sections": ["Company Overview", "Financial Health", "Recommendation"],
    },
]


def get_or_create_dataset(
    dataset_name: str = "financial-agent-evals",
    description: str = "Eval dataset for financial research agent",
) -> str:
    """Get existing dataset or create new one. Returns dataset name."""
    client = Client()

    # Check if dataset exists
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
        return dataset.name
    except Exception:
        pass

    # Create new dataset
    dataset = client.create_dataset(dataset_name=dataset_name, description=description)
    print(f"Created new dataset: {dataset_name}")

    # Add examples
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


def add_custom_ticker(dataset_name: str, ticker: str) -> None:
    """Add custom ticker to existing dataset."""
    client = Client()
    dataset = client.read_dataset(dataset_name=dataset_name)

    client.create_example(
        dataset_id=dataset.id,
        inputs={"ticker": ticker},
        outputs={
            "expected_tools": ["get_company_profile", "get_financial_ratios", "get_financial_statements"],
            "expected_sections": ["Company Overview", "Financial Health", "Recommendation"],
        },
    )
    print(f"Added ticker {ticker} to dataset")
