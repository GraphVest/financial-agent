from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.state import AgentState
from src.tools import get_company_profile, get_financial_ratios, get_financial_statements, get_stock_news

# Load env to get OPENAI_API_KEY
load_dotenv()

# --- 1. SETUP LLM & TOOLS ---
# Use GPT-4o-mini or GPT-3.5-turbo for speed and cost efficiency
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# List of tools we created in Phase 3
tools = [get_company_profile, get_financial_ratios, get_stock_news, get_financial_statements]

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
        3. get_stock_news
        
        Do not write the report yet. Just fetch the data.
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
    It takes all the gathered data (from tool outputs) and writes a final report.
    """
    messages = state["messages"]
    ticker = state["ticker"]

    # Create a prompt for the final report
    prompt = f"""
    You are a Senior Financial Analyst. 
    Based on the tool outputs above, write a comprehensive investment report for {ticker}.
    
    The report must be in Markdown format and include:
    1. **Company Overview**: What do they do?
    2. **Financial Health**: Analyze P/E, EPS, ROE, etc.
    3. **Market Sentiment**: Summarize the latest news.
    4. **Recommendation**: Buy, Hold, or Sell (based on your analysis).
    
    Be professional, concise, and data-driven.
    """

    # We use the raw LLM (without tools) for writing, as we just want text generation now.
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

# Tools -> Back to Researcher (to see if more data is needed)
workflow.add_edge("tools", "researcher")

# Writer -> End
workflow.add_edge("writer", END)

# Compile the graph
app = workflow.compile()
