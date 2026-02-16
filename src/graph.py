from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.state import AgentState
from src.tools import (
    get_analyst_estimates,
    get_company_profile,
    get_earnings_summary_via_search,
    get_financial_ratios,
    get_financial_statements,
    get_ownership_via_search,
    get_revenue_segmentation,
    # get_stock_news,
)

# Load env to get OPENAI_API_KEY
load_dotenv()

# --- 1. SETUP LLM & TOOLS ---
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# List of all available tools
tools = [
    get_company_profile,
    get_financial_ratios,
    get_financial_statements,
    # get_stock_news,
    get_earnings_summary_via_search,
    get_revenue_segmentation,
    get_analyst_estimates,
    get_ownership_via_search,
]

# Bind tools to the LLM. This gives the LLM the ability to "know" these tools exist.
llm_with_tools = llm.bind_tools(tools)


# --- 2. DEFINE NODES ---


def researcher_node(state: AgentState):
    """
    The Researcher Node.
    It looks at the state (ticker) and decides which tools to call.
    """
    messages = state["messages"]

    # If this is the first turn, we need to inject a system prompt
    # to guide the LLM to use tools.
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        ticker = state["ticker"]
        system_msg = SystemMessage(
            content=f"""
        You are a Wall Street Research Assistant. 
        Your goal is to gather comprehensive data for the ticker: {ticker}.
        
        You MUST call the following tools to get the data:
        1. get_company_profile
        2. get_financial_ratios
        3. get_financial_statements
        4. get_earnings_summary_via_search
        5. get_revenue_segmentation
        6. get_analyst_estimates
        7. get_ownership_via_search
        
        CRITICAL RULES:
        - ONLY call the tools listed above. Do NOT write any analysis, summary, or report.
        - Call ALL tools. Do NOT skip any.
        - After tools return data, respond with ONLY: "Data collection complete."
        - Do NOT summarize, interpret, or analyze the tool outputs.
        - Do NOT add any information from your internal knowledge.
        - The analysis and report writing will be handled by a separate writer.
        - **EXECUTE TOOLS IN PARALLEL to reduce latency.**
        """
        )
        # Prepend system message to history
        messages = [system_msg] + messages

    # Invoke the LLM (with tools bound)
    response = llm_with_tools.invoke(messages)

    # Return the new message (AIMessage) to update the state
    return {"messages": [response]}


def writer_node(state: AgentState):
    """
    The Writer Node.
    Focuses purely on content quality: analysis, storytelling, and actionable insights.
    Formatting and cleanup will be handled by the Publisher Node downstream.
    """
    messages = state["messages"]
    ticker = state["ticker"]

    prompt = f"""
    You are a Senior Chief Investment Officer (CIO) writing a premium analysis for serious investors.
    Your ONLY job is to write the best possible content — insightful, data-driven, and compelling.
    
    ---
    ### 🎨 WRITING STYLE (THE "COLSON" STYLE)
    Format every section header as: `[Standard Technical Term]: [A Catchy, Stock-Specific Insight]`

    * *Bad Example:* "2. Financial Analysis" (Too boring, too generic)
    * *Good Example (NVDA):* "2. Operating Leverage: Margins Hitting Software Levels"
    * *Good Example (TSLA):* "2. Valuation: Priced for Robotaxi Perfection?"
    * *Good Example (KO):* "2. The Moat: The World's Strongest Distribution Network"

    The part after the colon MUST be specific to {ticker}. Use product names, specific risks, or vivid metaphors found in the data.
    Use paragraphs for the main explanation, and only use bullet points for the specific data metrics.
    Ensure the tone is "Smart Mentor" — sophisticated but accessible. Avoid robotic transitions.

    ---
    ### 📝 REPORT STRUCTURE

    # {ticker} Deep Dive: [Create a Headline that summarizes the entire Bull/Bear thesis]

    **🚨 Executive Summary (TL;DR)**
    * **The Hook:** What is the one thing the market is missing or pricing in?
    * **The Numbers:** Quick summary of Revenue, Margins, and FCF.
    * **The Verdict:** Bullish, Bearish, or Neutral?

    **1. Business Transformation: [Insert Insight about the Pivot/Evolution]**
    * *Focus:* Don't just say what they do. Explain how the business mix has shifted (e.g., Gaming -> AI, or Hardware -> Services).
    * Use **Revenue Segmentation** data to prove the shift.

    **2. The Moat & Competitive Advantage: [Insert Insight about Why They Win]**
    * *Focus:* The "Secret Sauce".
    * **MANDATORY:** Mention specific products (e.g., Blackwell, CUDA, iPhone ecosystem) found in search.
    * Why is it hard for competitors to catch up?

    **3. Financial Performance: [Insert Insight about Margins or Cash Flow]**
    * *Focus:* Operating Leverage and Capital Allocation.
    * Interpret the **Income Statement**: Are they burning cash or printing it?
    * "They keep X cents on the dollar" analogy.

    **4. Outlook & Future Roadmap: [Insert Insight about the Next Big Thing]**
    * *Focus:* What is Management guiding? (Use Transcript info).
    * What is the next product launch that matters?

    **5. The Bear Case & Risks: [Insert Insight about the Biggest Fear]**
    * *Focus:* Concentration, Geopolitics, or Valuation bubbles.
    * Use **Institutional Ownership** to see if smart money is scared.

    **6. Valuation & The Verdict: [Insert Insight about Price vs. Value]**
    * *Format:* A concluding paragraph on whether it's priced for perfection.
    * *Data:* **MANDATORY:** You MUST cite the specific "Analyst Revenue Estimates" for the next year (e.g., 2026) found in the data.
    * *Actionable Scenarios (Safe Framing):* Identify specific "Technical Entry Zones" or "Allocation Scenarios" (e.g., "Historical support implies interest around $160-$168" or "A standard 3-5% position size fits this risk profile").
    * **CRITICAL CONSTRAINT:** Do NOT use command verbs like "Buy this," "Sell that." Instead, use phrases like "Investors might consider...", "Attractive risk/reward appears at...", "Technical levels suggest...".
    """

    response = llm.invoke(messages + [HumanMessage(content=prompt)])

    return {"messages": [response]}


def publisher_node(state: AgentState):
    """
    The Publisher Node.
    Post-processes the Writer's draft for clean, consistent formatting.
    Does NOT change content, wording, tone, or data — only presentation.
    """
    messages = state["messages"]

    prompt = """
    You are a meticulous Report Publisher. You have received a financial analysis draft.
    Your job is STRICTLY limited to formatting and cleanup. 
    
    You must OUTPUT the report with these formatting fixes applied:

    ### REMOVE (delete entirely):
    - Any introductory meta-text (e.g., "Here is the analysis you requested...", "Below is the report...")
    - Any closing remarks or follow-up offers (e.g., "Would you like me to...", "Let me know if...", "Feel free to ask...")
    - Any "Prepared by: ..." or signature lines
    - Any instructions, disclaimers that the AI added on its own (not part of the analysis)
    - The word "draft" if used to refer to the report itself

    ### FIX (adjust formatting only):
    - Flatten any nested bullet points into single-level bullets
    - Ensure consistent bullet style (use - for bullets)
    - Ensure one blank line between sections, no excessive blank lines
    - Ensure the report ends cleanly with "End of report." and nothing after

    ### DO NOT TOUCH:
    - The actual analysis content, wording, and tone
    - Any numbers, data points, or financial figures
    - Section headers and their order
    - The analytical conclusions and recommendations

    Output ONLY the cleaned report. No commentary before or after.
    """

    response = llm.invoke(messages + [HumanMessage(content=prompt)])

    return {"messages": [response]}


# --- 3. DEFINE CONDITIONAL EDGES ---


def should_continue(state: AgentState) -> Literal["tools", "writer"]:
    """
    Decides the next step:
    - If the LLM made tool calls -> Go to 'tools' node to execute them.
    - If the LLM has finished gathering data (no tool calls) -> Go to 'writer' node.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM wants to call tools (tool_calls attribute is not empty)
    if last_message.tool_calls:
        return "tools"

    # Otherwise, assume data gathering is done, move to writing
    return "writer"


# --- 4. BUILD THE GRAPH ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", ToolNode(tools))  # ToolNode is a prebuilt node from LangGraph
workflow.add_node("writer", writer_node)
workflow.add_node("publisher", publisher_node)

# Define the flow
# Start -> Researcher
workflow.add_edge(START, "researcher")

# Researcher -> (Check condition) -> Tools OR Writer
workflow.add_conditional_edges(
    "researcher",
    should_continue,
    {
        "tools": "tools",  # If 'tools', go to 'tools' node
        "writer": "writer",  # If 'writer', go to 'writer' node
    },
)

# Tools -> Go directly to Writer (skip researcher's intermediate summary)
workflow.add_edge("tools", "writer")

# Writer -> Publisher -> End
workflow.add_edge("writer", "publisher")
workflow.add_edge("publisher", END)

# Compile the graph
app = workflow.compile()
