"""
Custom evaluators for financial agent evaluation.
Includes: Faithfulness, Completeness, Tool Coverage
"""

import re

from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run


# --- 1. FAITHFULNESS EVALUATOR (LLM-as-Judge) ---
def faithfulness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Check if final output only uses data from tool outputs (no hallucination).
    Uses LLM-as-Judge approach.
    """
    # Extract final output (last message content)
    final_output = _get_final_output(run)
    if not final_output:
        return EvaluationResult(key="faithfulness", score=0.0, comment="No output found")

    # Extract tool outputs from run
    tool_outputs = _get_tool_outputs(run)
    if not tool_outputs:
        return EvaluationResult(key="faithfulness", score=0.5, comment="No tool outputs to compare")

    # LLM-as-Judge
    judge = ChatOpenAI(model="gpt-5-mini", temperature=0)
    prompt = f"""You are evaluating if a financial report ONLY uses data from provided tool outputs.

TOOL OUTPUTS (ground truth data):
{tool_outputs}

REPORT TO EVALUATE:
{final_output}

TASK:
1. Check if ALL claims/numbers in the report come from tool outputs
2. Flag any information that appears fabricated or not in tool outputs

SCORING:
- 1.0: All info comes from tool outputs, no hallucination
- 0.7: Minor additions but core facts correct  
- 0.5: Some unverifiable claims
- 0.3: Significant fabrication
- 0.0: Mostly hallucinated

Respond with ONLY a JSON: {{"score": <float>, "reason": "<brief explanation>"}}"""

    try:
        response = judge.invoke(prompt)
        content = response.content.strip()
        # Parse JSON from response
        match = re.search(r'\{[^}]+\}', content)
        if match:
            import json
            result = json.loads(match.group())
            return EvaluationResult(
                key="faithfulness",
                score=float(result.get("score", 0)),
                comment=result.get("reason", ""),
            )
    except Exception as e:
        return EvaluationResult(key="faithfulness", score=0.0, comment=f"Eval error: {e}")

    return EvaluationResult(key="faithfulness", score=0.0, comment="Failed to parse judge response")


# --- 2. COMPLETENESS EVALUATOR (Rule-based) ---
def completeness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Check if report contains all required sections.
    Rule-based: checks for section headers.
    """
    final_output = _get_final_output(run)
    if not final_output:
        return EvaluationResult(key="completeness", score=0.0, comment="No output found")

    expected_sections = example.outputs.get("expected_sections", [])
    if not expected_sections:
        expected_sections = ["Company Overview", "Financial Health", "Recommendation"]

    found = []
    missing = []

    for section in expected_sections:
        # Check various header formats
        patterns = [
            section.lower(),
            section.replace(" ", "").lower(),
            f"## {section}".lower(),
            f"**{section}**".lower(),
        ]
        if any(p in final_output.lower() for p in patterns):
            found.append(section)
        else:
            missing.append(section)

    score = len(found) / len(expected_sections) if expected_sections else 1.0
    comment = f"Found: {found}" if score == 1.0 else f"Missing: {missing}"

    return EvaluationResult(key="completeness", score=score, comment=comment)


# --- 3. TOOL COVERAGE EVALUATOR (Rule-based) ---
def tool_coverage_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Check if all expected tools were called.
    Rule-based: checks run child_runs for tool executions.
    """
    expected_tools = example.outputs.get("expected_tools", [])
    if not expected_tools:
        expected_tools = ["get_company_profile", "get_financial_ratios", "get_financial_statements"]

    called_tools = _get_called_tools(run)

    found = [t for t in expected_tools if t in called_tools]
    missing = [t for t in expected_tools if t not in called_tools]

    score = len(found) / len(expected_tools) if expected_tools else 1.0
    comment = f"Called: {found}" if score == 1.0 else f"Missing: {missing}"

    return EvaluationResult(key="tool_coverage", score=score, comment=comment)


# --- HELPER FUNCTIONS ---
def _get_final_output(run: Run) -> str:
    """Extract final output string from run."""
    if run.outputs:
        # Try common output keys
        for key in ["output", "messages", "content", "result"]:
            if key in run.outputs:
                val = run.outputs[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, list) and val:
                    # Get last message content
                    last = val[-1]
                    if hasattr(last, "content"):
                        return last.content
                    if isinstance(last, dict):
                        return last.get("content", str(last))
                    return str(last)
        return str(run.outputs)
    return ""


def _get_tool_outputs(run: Run) -> str:
    """Extract tool outputs from run's messages."""
    tool_outputs = []

    # First try child_runs (LangSmith trace)
    if hasattr(run, "child_runs") and run.child_runs:
        for child in run.child_runs:
            if child.run_type == "tool":
                name = child.name or "unknown_tool"
                output = child.outputs.get("output", "") if child.outputs else ""
                tool_outputs.append(f"[{name}]: {output}")

    # If no child_runs, extract from messages
    if not tool_outputs and run.outputs:
        messages = run.outputs.get("messages", [])
        for msg in messages:
            # Check for ToolMessage type
            if hasattr(msg, "type") and msg.type == "tool":
                name = getattr(msg, "name", "unknown_tool")
                content = getattr(msg, "content", "")
                tool_outputs.append(f"[{name}]: {content}")
            # Check for dict format
            elif isinstance(msg, dict) and msg.get("type") == "ToolMessage":
                name = msg.get("tool_name", msg.get("name", "unknown"))
                content = msg.get("content", "")
                tool_outputs.append(f"[{name}]: {content}")

    return "\n".join(tool_outputs) if tool_outputs else ""


def _get_called_tools(run: Run) -> list[str]:
    """Get list of tool names called in this run."""
    tools = []

    # First try child_runs (LangSmith trace)
    if hasattr(run, "child_runs") and run.child_runs:
        for child in run.child_runs:
            if child.run_type == "tool" and child.name:
                tools.append(child.name)

    # If no child_runs, extract from messages
    if not tools and run.outputs:
        messages = run.outputs.get("messages", [])
        for msg in messages:
            # Check for AIMessage with tool_calls attribute
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name", "")
                        if name:
                            tools.append(name)
                    elif hasattr(tc, "name"):
                        tools.append(tc.name)

            # Check for ToolMessage (class name check)
            msg_type = type(msg).__name__
            if msg_type == "ToolMessage":
                name = getattr(msg, "name", None)
                if name:
                    tools.append(name)

            # Check for dict format with tool_calls
            if isinstance(msg, dict):
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        name = tc.get("name", "")
                        if name:
                            tools.append(name)

    return list(set(tools))  # Dedupe


# Export all evaluators
EVALUATORS = [
    faithfulness_evaluator,
    completeness_evaluator,
    tool_coverage_evaluator,
]
