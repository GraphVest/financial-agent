"""
Langfuse-based evaluation runner for financial agent.
Usage: python -m eval.langfuse_runner --ticker AAPL
       python -m eval.langfuse_runner --batch AAPL NVDA MSFT
"""

import argparse
import asyncio
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse import get_client
from langsmith.schemas import Example, Run

from eval.datasets import EXPECTED_SECTIONS, EXPECTED_TOOLS
from eval.evaluators import (
    completeness_evaluator,
    faithfulness_evaluator,
    tool_coverage_evaluator,
)
from src.graph import app

load_dotenv()


async def run_agent(ticker: str) -> dict:
    """Run financial agent for a ticker."""
    initial_state = {
        "messages": [HumanMessage(content=f"Research {ticker} stock.")],
        "ticker": ticker,
    }

    result = await app.ainvoke(initial_state)
    messages = result.get("messages", [])
    final_output = messages[-1].content if messages else ""

    return {"output": final_output, "messages": messages}


def run_langfuse_eval(ticker: str, dataset_name: str = "financial-agent") -> dict:
    """
    Run evaluation with Langfuse scoring.
    Logs trace + scores to Langfuse dashboard.
    """
    print(f"\n{'='*50}")
    print(f"Running Langfuse eval for: {ticker}")
    print(f"{'='*50}\n")

    # Get Langfuse client (singleton)
    langfuse = get_client()

    # Check auth
    if not langfuse.auth_check():
        print("ERROR: Langfuse authentication failed. Check your credentials.")
        return {}

    # Create a span that encompasses the entire evaluation
    with langfuse.start_as_current_span(name=f"eval-{ticker}") as span:
        # Update span with metadata
        span.update(
            metadata={"ticker": ticker, "dataset": dataset_name},
            tags=["evaluation", "financial-agent"],
        )

        # Run agent inside the span context
        with langfuse.start_as_current_span(name="agent-run") as agent_span:
            output = asyncio.run(run_agent(ticker))
            agent_span.update(output=output)

        # Create mock objects for evaluators (reuse existing logic)
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
                "expected_tools": EXPECTED_TOOLS,
                "expected_sections": EXPECTED_SECTIONS,
            },
            created_at=datetime.now(timezone.utc),
        )

        # Run evaluators and log scores to Langfuse
        results = {}
        evaluators = [
            ("faithfulness", faithfulness_evaluator),
            ("completeness", completeness_evaluator),
            ("tool_coverage", tool_coverage_evaluator),
        ]

        for name, evaluator in evaluators:
            result = evaluator(mock_run, mock_example)
            results[name] = {"score": result.score, "comment": result.comment}

            # Log score to Langfuse - now inside active span context
            langfuse.score_current_trace(
                name=name,
                value=result.score,
                comment=result.comment,
            )
            print(f"  {name}: {result.score:.2f} - {result.comment}")

        # Get trace URL while still in context
        trace_url = langfuse.get_trace_url()

    # Flush to ensure all data is sent
    langfuse.flush()

    if trace_url:
        print(f"\nView trace at: {trace_url}")
    else:
        print("\nView results at: https://cloud.langfuse.com")

    return results


def run_langfuse_batch(tickers: list[str], dataset_name: str = "financial-agent") -> dict:
    """Run batch evaluation on multiple tickers."""
    all_results = {}

    for ticker in tickers:
        all_results[ticker] = run_langfuse_eval(ticker, dataset_name)

    # Summary
    print(f"\n{'='*50}")
    print("BATCH SUMMARY")
    print(f"{'='*50}")

    for metric in ["faithfulness", "completeness", "tool_coverage"]:
        scores = [r[metric]["score"] for r in all_results.values() if r]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{metric}: {avg:.2f} avg")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Langfuse evaluations")
    parser.add_argument("--ticker", help="Single ticker to evaluate")
    parser.add_argument("--batch", nargs="+", help="Multiple tickers for batch eval")

    args = parser.parse_args()

    if args.batch:
        run_langfuse_batch(args.batch)
    elif args.ticker:
        run_langfuse_eval(args.ticker)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

