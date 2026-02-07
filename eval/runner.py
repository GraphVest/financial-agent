"""
Main eval runner for financial agent.
Usage: python -m eval.runner [--dataset NAME] [--tickers AAPL,NVDA]
"""

import argparse
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langsmith.evaluation import evaluate

from eval.datasets import get_or_create_dataset
from eval.evaluators import EVALUATORS
from src.graph import app

load_dotenv()


async def run_agent(inputs: dict) -> dict:
    """
    Target function for evaluation.
    Takes inputs from dataset and returns agent output.
    """
    ticker = inputs["ticker"]

    initial_state = {
        "messages": [HumanMessage(content=f"Research {ticker} stock.")],
        "ticker": ticker,
    }

    # Run agent
    result = await app.ainvoke(initial_state)

    # Extract final output
    messages = result.get("messages", [])
    final_output = messages[-1].content if messages else ""

    return {"output": final_output, "messages": messages}


def run_agent_sync(inputs: dict) -> dict:
    """Sync wrapper for async agent."""
    return asyncio.run(run_agent(inputs))


def run_evaluation(
    dataset_name: str = "financial-agent-evals",
    experiment_prefix: str = "financial-agent",
) -> dict:
    """
    Run evaluation on dataset.
    Returns evaluation results.
    """
    # Ensure dataset exists
    get_or_create_dataset(dataset_name)

    print(f"\n{'='*50}")
    print(f"Running evaluation on dataset: {dataset_name}")
    print(f"{'='*50}\n")

    # Run eval
    results = evaluate(
        target=run_agent_sync,
        data=dataset_name,
        evaluators=EVALUATORS,
        experiment_prefix=experiment_prefix,
    )

    print(f"\n{'='*50}")
    print("Evaluation complete!")
    print("View results at: https://smith.langchain.com")
    print(f"{'='*50}\n")

    return results


def run_single_eval(ticker: str) -> dict:
    """
    Run evaluation on single ticker (for quick testing).
    """
    import uuid
    from datetime import datetime, timezone

    print(f"\n{'='*50}")
    print(f"Running single eval for: {ticker}")
    print(f"{'='*50}\n")

    # Run agent
    output = run_agent_sync({"ticker": ticker})
    print(f"Agent output length: {len(output.get('output', ''))}")

    # Create mock run/example for evaluators with valid UUIDs
    from langsmith.schemas import Example, Run

    run_id = uuid.uuid4()
    mock_run = Run(
        id=run_id,
        name="financial-agent-test",
        run_type="chain",
        inputs={"ticker": ticker},
        outputs=output,
        start_time=datetime.now(timezone.utc),
        trace_id=run_id,
    )

    mock_example = Example(
        id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        inputs={"ticker": ticker},
        outputs={
            "expected_tools": ["get_company_profile", "get_financial_ratios", "get_financial_statements"],
            "expected_sections": ["Company Overview", "Financial Health", "Recommendation"],
        },
        created_at=datetime.now(timezone.utc),
    )

    # Run evaluators
    results = {}
    for evaluator in EVALUATORS:
        result = evaluator(mock_run, mock_example)
        results[result.key] = {"score": result.score, "comment": result.comment}
        print(f"  {result.key}: {result.score:.2f} - {result.comment}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run financial agent evaluations")
    parser.add_argument("--dataset", default="financial-agent-evals", help="Dataset name")
    parser.add_argument("--ticker", help="Single ticker for quick test (skips dataset)")
    parser.add_argument("--prefix", default="financial-agent", help="Experiment prefix")

    args = parser.parse_args()

    if args.ticker:
        # Quick single-ticker test
        run_single_eval(args.ticker)
    else:
        # Full dataset eval
        run_evaluation(dataset_name=args.dataset, experiment_prefix=args.prefix)


if __name__ == "__main__":
    main()
