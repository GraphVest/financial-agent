import asyncio

from src.tools import get_company_profile, get_financial_ratios, get_stock_news


async def main():
    ticker = "NVDA"  # Nvidia
    print(f"--- TESTING TOOLS FOR {ticker} ---")

    # 1. Test Profile Tool
    print("\n[1] Invoking 'get_company_profile'...")
    # Note: We simulate how the Agent calls the tool (passing a dict)
    profile_result = await get_company_profile.ainvoke({"ticker": ticker})
    print(f"Result: {profile_result}")

    # 2. Test Metrics Tool
    print("\n[2] Invoking 'get_financial_ratios'...")
    metrics_result = await get_financial_ratios.ainvoke({"ticker": ticker})
    print(f"Result: {metrics_result}")

    # 3. Test News Tool
    print("\n[3] Invoking 'get_stock_news'...")
    news_result = await get_stock_news.ainvoke({"ticker": ticker})

    # Pretty print just the first news title to keep terminal clean
    if isinstance(news_result, list) and len(news_result) > 0 and "title" in news_result[0]:
        print(f"First News Title: {news_result[0]['title']}")
    else:
        print(f"Result: {news_result}")


if __name__ == "__main__":
    asyncio.run(main())
