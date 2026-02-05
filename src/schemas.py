from typing import Optional

from pydantic import BaseModel, Field


# 1. Company Profile Model
class StockProfile(BaseModel):
    """
    Represents the company profile data.
    """

    symbol: str
    company_name: str = Field(alias="companyName")
    price: float
    market_cap: float = Field(alias="marketCap")
    description: str
    sector: str
    industry: str
    ceo: Optional[str] = None
    website: Optional[str] = None

    # Allow extra fields in case API returns more data than defined
    model_config = {"extra": "ignore"}


# 2. Financial Metrics Model (Key Ratios)
class KeyMetrics(BaseModel):
    """
    Represents key financial ratios (TTM - Trailing Twelve Months).
    Uses stable API ratios-ttm endpoint field names.
    """

    symbol: str
    pe_ratio: Optional[float] = Field(alias="priceToEarningsRatioTTM", default=None)
    eps: Optional[float] = Field(alias="netIncomePerShareTTM", default=None)
    roe: Optional[float] = Field(alias="returnOnEquityTTM", default=None)
    debt_to_equity: Optional[float] = Field(alias="debtEquityRatioTTM", default=None)
    dividend_yield: Optional[float] = Field(alias="dividendYieldTTM", default=None)

    model_config = {"extra": "ignore", "populate_by_name": True}


# 3. Market News Model
class MarketNews(BaseModel):
    """
    Represents a single news article.
    """

    title: str
    date: str = Field(alias="publishedDate")
    site: str
    text: str
    url: str
    sentiment: Optional[str] = None  # FMP sometimes provides sentiment analysis
    image: Optional[str] = None

    model_config = {"extra": "ignore"}


class FinancialStatement(BaseModel):
    """
    Generic model for Income Statement, Balance Sheet, or Cash Flow.
    We use extra='allow' because these statements have 50+ dynamic fields.
    """

    date: str
    symbol: str
    reportedCurrency: str = "USD"
    fillingDate: Optional[str] = None
    period: str

    # Allow all other fields (revenue, netIncome, totalAssets, etc.) to pass through
    model_config = {"extra": "allow"}
