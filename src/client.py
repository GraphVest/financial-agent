import os
from typing import Any, List, Optional

import httpx
from dotenv import load_dotenv

from src.schemas import KeyMetrics, MarketNews, StockProfile

# Load environment variables from .env file
load_dotenv()


class FMPClient:
    """
    Async client for the Financial Modeling Prep (FMP) API.
    Includes fallback logic for restricted/legacy endpoints on new accounts.
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY environment variable is not set")

        # Initialize async client with a timeout
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Closes the underlying HTTP client session."""
        await self.client.aclose()

    async def _get(self, endpoint: str, params: dict = {}) -> Any:
        """
        Internal helper method to execute GET requests with error handling.
        """
        params = params.copy()
        params["apikey"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()  # Raises exception for 4xx/5xx codes
            return response.json()

        except httpx.HTTPStatusError as e:
            # Print detailed error from FMP (e.g., "Legacy Endpoint")
            print(f"❌ HTTP Error {e.response.status_code} at endpoint '{endpoint}': {e.response.text}")
            return []
        except Exception as e:
            print(f"❌ Request Error: {e}")
            return []

    async def get_profile(self, ticker: str) -> Optional[StockProfile]:
        """
        Fetches company profile.
        FALLBACK STRATEGY:
        If the standard 'profile' endpoint returns 403 (Legacy User restriction),
        it attempts to fetch basic data from the 'quote' endpoint instead.
        """
        # 1. Try the standard profile endpoint (stable API uses query params)
        endpoint = "profile"
        data = await self._get(endpoint, {"symbol": ticker})

        if isinstance(data, list) and len(data) > 0:
            return StockProfile(**data[0])

        # 2. Fallback: Try 'quote' endpoint if profile failed or returned empty
        print("⚠️  'Profile' endpoint failed or empty. Attempting fallback to 'Quote'...")
        quote_endpoint = "quote"
        quote_data = await self._get(quote_endpoint, {"symbol": ticker})

        if isinstance(quote_data, list) and len(quote_data) > 0:
            q = quote_data[0]
            # Manual mapping from Quote data to StockProfile model
            return StockProfile(
                symbol=q.get("symbol"),
                companyName=q.get("name"),  # 'quote' uses 'name', 'profile' uses 'companyName'
                price=q.get("price"),
                mktCap=q.get("marketCap"),
                description="Description unavailable (Source: Quote Endpoint)",
                sector="N/A",
                industry="N/A",
                ceo="N/A",
                website="N/A",
            )

        return None

    async def get_key_metrics(self, ticker: str) -> Optional[KeyMetrics]:
        """
        Fetches key financial ratios (TTM) using stable API.
        Uses ratios-ttm endpoint which includes PE ratio and EPS.
        """
        endpoint = "ratios-ttm"
        data = await self._get(endpoint, {"symbol": ticker})

        if isinstance(data, list) and len(data) > 0:
            return KeyMetrics(**data[0])
        return None

    async def get_news(self, ticker: str, limit: int = 5) -> List[MarketNews]:
        """
        Fetches stock market news using stable API.
        """
        endpoint = "news/stock"
        params = {"symbols": ticker, "limit": limit}
        data = await self._get(endpoint, params)

        news_list = []
        if isinstance(data, list):
            for item in data:
                # Validate and append only valid news items
                try:
                    news_list.append(MarketNews(**item))
                except Exception:
                    # Skip items that don't match the schema
                    continue

        return news_list

    async def get_financial_statements(self, ticker: str, statement_type: str, limit: int = 4) -> List[dict]:
        """
        Fetches financial statements (Income, Balance Sheet, Cash Flow).

        Args:
            ticker: Stock symbol.
            statement_type: One of 'income-statement', 'balance-sheet-statement', 'cash-flow-statement'.
            limit: Number of periods to fetch (default 4 years).
        """
        # Endpoint format: /v3/income-statement/AAPL?limit=4
        endpoint = f"{statement_type}/{ticker}"
        params = {"limit": limit}

        # Note: Financial statements usually don't have a 403 restriction on stable API,
        # but if they do, we return an empty list.
        data = await self._get(endpoint, params)

        if isinstance(data, list):
            return data
        return []
