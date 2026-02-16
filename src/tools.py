import asyncio
from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src.client import FMPClient
from src.schemas import KeyMetrics, MarketNews, StockProfile


# --- 1. Define Input Schemas (What the LLM sends to the tool) ---
class TickerInput(BaseModel):
    """Input schema for stock ticker operations."""

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL, TSLA, NVDA).")


# --- 2. Define Tools (The functions the LLM can call) ---


@tool("get_company_profile", args_schema=TickerInput)
async def get_company_profile(ticker: str) -> dict[str, Any]:
    """
    Fetches the company profile (CEO, description, sector, price, market cap).
    Useful for understanding what the company does and its general standing.
    """
    client = FMPClient()
    try:
        profile: StockProfile | None = await client.get_profile(ticker)
        if profile:
            # Return model as dictionary for the LLM to read
            return profile.model_dump(by_alias=True)
        return {"error": f"Company profile not found for ticker: {ticker}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


@tool("get_financial_ratios", args_schema=TickerInput)
async def get_financial_ratios(ticker: str) -> dict[str, Any]:
    """
    Fetches key financial ratios (PE, EPS, ROE, Debt/Equity) for the trailing twelve months (TTM).
    Useful for evaluating valuation and profitability.
    """
    client = FMPClient()
    try:
        metrics: KeyMetrics | None = await client.get_key_metrics(ticker)
        if metrics:
            return metrics.model_dump(by_alias=True)
        return {"error": f"Financial metrics not found for ticker: {ticker}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


@tool("get_stock_news", args_schema=TickerInput)
async def get_stock_news(ticker: str) -> list[dict[str, Any]]:
    """
    Fetches the latest market news related to the stock.
    Useful for sentiment analysis and identifying recent events.
    """
    client = FMPClient()
    try:
        news_list: list[MarketNews] | None = await client.get_news(ticker, limit=5)
        if news_list:
            return [news.model_dump(by_alias=True) for news in news_list]
        return [{"error": "No recent news found."}]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        await client.close()


@tool("get_financial_statements", args_schema=TickerInput)
async def get_financial_statements(ticker: str) -> dict[str, Any]:
    """
    Fetches the Income Statement, Balance Sheet, and Cash Flow Statement for the last 4 years.
    CRITICAL for valuation (DCF), margin analysis, and calculating growth rates.
    """
    client = FMPClient()
    try:
        # Execute 3 API calls in parallel to reduce latency
        income, balance, cash = await asyncio.gather(
            client.get_financial_statements(ticker, "income-statement"),
            client.get_financial_statements(ticker, "balance-sheet-statement"),
            client.get_financial_statements(ticker, "cash-flow-statement"),
        )

        if income or balance or cash:
            return {
                "income_statement": [stmt.model_dump(by_alias=True) for stmt in income] if income else [],
                "balance_sheet": [stmt.model_dump(by_alias=True) for stmt in balance] if balance else [],
                "cash_flow": [stmt.model_dump(by_alias=True) for stmt in cash] if cash else [],
            }
        return {"error": f"Financial statements not found for ticker: {ticker}"}
        
    except Exception as e:
        return {"error": f"Failed to fetch financials: {str(e)}"}
    finally:
        await client.close()

# --- COMMENTED OUT: Using Tavily search alternatives to reduce FMP API costs ---
# @tool("get_earnings_transcript", args_schema=TickerInput)
# async def get_earnings_transcript(ticker: str) -> dict[str, Any]:
#     """
#     Fetches the most recent earnings call transcript.
#     CRITICAL for qualitative analysis. This is where you find:
#     1. Management guidance and future outlook.
#     2. Specific product mentions (e.g., 'Blackwell', 'Rubin').
#     3. Explanations for margin changes or strategic shifts.
#     """
#     client = FMPClient()
#     try:
#         dates = await client.get_transcript_dates(ticker)
#         if not dates:
#             return {"error": "No earnings transcript dates found."}
#         latest = dates[0]
#         year = latest.get("year") or latest.get("fiscalYear")
#         quarter = latest.get("quarter")
#         if not year or not quarter:
#             return {"error": "Could not determine latest transcript year/quarter."}
#         transcripts = await client.get_transcript(ticker, year=int(year), quarter=int(quarter))
#         if transcripts:
#             t = transcripts[0]
#             return {
#                 "date": t.get("date"),
#                 "quarter": t.get("quarter"),
#                 "year": t.get("year"),
#                 "content": t.get("content"),
#             }
#         return {"error": "No earnings transcripts found."}
#     except Exception as e:
#         return {"error": str(e)}
#     finally:
#         await client.close()


@tool("get_revenue_segmentation", args_schema=TickerInput)
async def get_revenue_segmentation(ticker: str) -> dict[str, Any]:
    """
    Fetches revenue breakdown by Product and Geography.
    ESSENTIAL for 'Business Analysis':
    - Product segments: Shows which division drives growth (e.g., Data Center vs. Gaming).
    - Geographic segments: Reveals exposure to specific regions (e.g., China risk).
    """
    client = FMPClient()
    try:
        # Fetch both breakdowns in parallel
        product, geo = await asyncio.gather(
            client.get_revenue_product_segmentation(ticker),
            client.get_revenue_geographic_segmentation(ticker)
        )
        
        return {
            "product_segments": product if product else [],
            "geographic_segments": geo if geo else []
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


@tool("get_analyst_estimates", args_schema=TickerInput)
async def get_analyst_estimates(ticker: str) -> list[dict[str, Any]]:
    """
    Fetches Wall Street consensus estimates for Revenue and EPS for upcoming quarters/years.
    REQUIRED for 'Valuation Context'.
    Allows comparison of current stock price against future growth expectations (Forward P/E).
    """
    client = FMPClient()
    try:
        # Fetch estimates for the next 5 periods
        estimates = await client.get_analyst_estimates(ticker, limit=5)
        if estimates:
            return [est.model_dump(by_alias=True) for est in estimates]
        return [{"error": "No analyst estimates found."}]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        await client.close()


# --- COMMENTED OUT: Using Tavily search alternative to reduce FMP API costs ---
# @tool("get_institutional_holders", args_schema=TickerInput)
# async def get_institutional_holders(ticker: str) -> list[dict[str, Any]]:
#     """
#     Fetches the top institutional holders and their share percentages.
#     Useful for 'Ownership Structure' analysis (Smart Money flow).
#     """
#     client = FMPClient()
#     try:
#         holders = await client.get_institutional_holders(ticker)
#         if holders:
#             return [holder.model_dump(by_alias=True) for holder in holders]
#         return [{"error": "No institutional holders data found."}]
#     except Exception as e:
#         return [{"error": str(e)}]
#     finally:
#         await client.close()


# --- 3. Tavily Search Tools (Cost-effective alternatives) ---

tavily_tool = TavilySearch(max_results=3, topic="finance")


@tool("get_earnings_summary_via_search", args_schema=TickerInput)
async def get_earnings_summary_via_search(ticker: str) -> str:
    """
    Searches for the latest earnings call summary, management guidance, and strategic updates.
    Provides qualitative analysis including management outlook, key takeaways, and future guidance.
    """
    query = f"{ticker} latest earnings call transcript summary key takeaways management guidance future outlook"
    try:
        response = await tavily_tool.ainvoke({"query": query})
        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            return "No earnings summary found."
        sections = []
        for r in results:
            sections.append(f"**Source:** {r.get('url', 'N/A')}\n{r.get('content', '')}")
        return "\n\n---\n\n".join(sections)
    except Exception as e:
        return f"Search failed: {e}"


@tool("get_ownership_via_search", args_schema=TickerInput)
async def get_ownership_via_search(ticker: str) -> str:
    """
    Searches for major institutional holders and ownership structure.
    Useful for Smart Money flow and institutional confidence analysis.
    """
    query = f"{ticker} top institutional holders ownership structure percentage shares"
    try:
        response = await tavily_tool.ainvoke({"query": query})
        results = response.get("results", []) if isinstance(response, dict) else []
        if not results:
            return "No ownership data found."
        sections = []
        for r in results:
            sections.append(f"**Source:** {r.get('url', 'N/A')}\n{r.get('content', '')}")
        return "\n\n---\n\n".join(sections)
    except Exception as e:
        return f"Search failed: {e}"