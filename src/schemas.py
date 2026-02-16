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
    ceo: str | None = None
    website: str | None = None

    # Allow extra fields in case API returns more data than defined
    model_config = {"extra": "ignore"}


# 2. Financial Metrics Model (Key Ratios)
class KeyMetrics(BaseModel):
    """
    Represents key financial ratios (TTM - Trailing Twelve Months).
    Uses stable API ratios-ttm endpoint field names.
    """

    symbol: str
    pe_ratio: float | None = Field(alias="priceToEarningsRatioTTM", default=None)
    eps: float | None = Field(alias="netIncomePerShareTTM", default=None)
    roe: float | None = Field(alias="returnOnEquityTTM", default=None)
    debt_to_equity: float | None = Field(alias="debtToEquityRatioTTM", default=None)
    dividend_yield: float | None = Field(alias="dividendYieldTTM", default=None)

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
    sentiment: str | None = None  # FMP sometimes provides sentiment analysis
    image: str | None = None

    model_config = {"extra": "ignore"}


class FinancialStatement(BaseModel):
    """
    Generic model for Income Statement, Balance Sheet, or Cash Flow.
    We use extra='allow' because these statements have 50+ dynamic fields.
    """

    date: str
    symbol: str
    reportedCurrency: str = "USD"
    fillingDate: str | None = None
    period: str

    # Allow all other fields (revenue, netIncome, totalAssets, etc.) to pass through
    model_config = {"extra": "allow"}


# 5. Analyst Estimates Model
class AnalystEstimate(BaseModel):
    """Analyst consensus estimates for revenue and EPS."""

    symbol: str
    date: str
    estimated_revenue_avg: float | None = Field(alias="revenueAvg", default=None)
    estimated_revenue_high: float | None = Field(alias="revenueHigh", default=None)
    estimated_revenue_low: float | None = Field(alias="revenueLow", default=None)
    estimated_eps_avg: float | None = Field(alias="epsAvg", default=None)
    estimated_eps_high: float | None = Field(alias="epsHigh", default=None)
    estimated_eps_low: float | None = Field(alias="epsLow", default=None)
    number_analyst_estimated_revenue: int | None = Field(
        alias="numAnalystsRevenue", default=None
    )
    number_analysts_estimated_eps: int | None = Field(
        alias="numAnalystsEps", default=None
    )

    model_config = {"extra": "ignore", "populate_by_name": True}


# 6. Institutional Holder Model (Form 13F)
class InstitutionalHolder(BaseModel):
    """An institutional holder from Form 13F filings."""

    investor_name: str = Field(alias="investorName")
    symbol: str
    security_name: str | None = Field(alias="securityName", default=None)
    shares_number: int | None = Field(alias="sharesNumber", default=None)
    market_value: float | None = Field(alias="marketValue", default=None)
    avg_price_paid: float | None = Field(alias="avgPricePaid", default=None)
    ownership_percent: float | None = Field(alias="ownershipPercent", default=None)
    change_in_shares_number_percentage: float | None = Field(
        alias="changeInSharesNumberPercentage", default=None
    )
    is_new: bool | None = Field(alias="isNew", default=None)

    model_config = {"extra": "ignore", "populate_by_name": True}
