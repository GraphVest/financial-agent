import asyncio
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.client import FMPClient
from src.schemas import KeyMetrics, MarketNews, StockProfile


# --- 1. Define Input Schemas (What the LLM sends to the tool) ---
class TickerInput(BaseModel):
    """Input schema for stock ticker operations."""

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL, TSLA, NVDA).")


# --- 2. Define Tools (The functions the LLM can call) ---


@tool("get_company_profile", args_schema=TickerInput)
async def get_company_profile(ticker: str) -> Dict[str, Any]:
    """
    Fetches the company profile (CEO, description, sector, price, market cap).
    Useful for understanding what the company does and its general standing.
    """
    client = FMPClient()
    try:
        profile: Optional[StockProfile] = await client.get_profile(ticker)
        if profile:
            # Return model as dictionary for the LLM to read
            return profile.model_dump(by_alias=True)
        return {"error": f"Company profile not found for ticker: {ticker}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


@tool("get_financial_ratios", args_schema=TickerInput)
async def get_financial_ratios(ticker: str) -> Dict[str, Any]:
    """
    Fetches key financial ratios (PE, EPS, ROE, Debt/Equity) for the trailing twelve months (TTM).
    Useful for evaluating valuation and profitability.
    """
    client = FMPClient()
    try:
        metrics: Optional[KeyMetrics] = await client.get_key_metrics(ticker)
        if metrics:
            return metrics.model_dump(by_alias=True)
        return {"error": f"Financial metrics not found for ticker: {ticker}"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        await client.close()


@tool("get_stock_news", args_schema=TickerInput)
async def get_stock_news(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetches the latest market news related to the stock.
    Useful for sentiment analysis and identifying recent events.
    """
    client = FMPClient()
    try:
        news_list: List[MarketNews] = await client.get_news(ticker, limit=5)
        if news_list:
            return [news.model_dump(by_alias=True) for news in news_list]
        return [{"error": "No recent news found."}]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        await client.close()


# --- ADD THIS NEW TOOL ---


@tool("get_financial_statements", args_schema=TickerInput)
async def get_financial_statements(ticker: str) -> Dict[str, Any]:
    """
    Fetches the Income Statement, Balance Sheet, and Cash Flow Statement for the last 4 years.
    CRITICAL for valuation (DCF), margin analysis, and calculating growth rates.
    """
    client = FMPClient()
    try:
        print(f"   --> 📊 Fetching deep financials for {ticker}...")

        # Execute 3 API calls in parallel to reduce latency
        income, balance, cash = await asyncio.gather(
            client.get_financial_statements(ticker, "income-statement"),
            client.get_financial_statements(ticker, "balance-sheet-statement"),
            client.get_financial_statements(ticker, "cash-flow-statement"),
        )

        # Return a structured dictionary
        return {"income_statement": income, "balance_sheet": balance, "cash_flow": cash}
    except Exception as e:
        return {"error": f"Failed to fetch financials: {str(e)}"}
    finally:
        await client.close()
