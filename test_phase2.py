import asyncio

from src.client import FMPClient


async def main():
    ticker = "AAPL"
    print(f"--- Fetching data for {ticker} ---")

    client = FMPClient()

    try:
        # 1. Test Profile
        print("\n[1] Fetching Profile...")
        profile = await client.get_profile(ticker)
        if profile:
            print(
                f"Success: {profile.company_name} | CEO: {profile.ceo} | Price: ${profile.price}"
            )
        else:
            print("Failed to fetch profile.")

        # 2. Test Metrics
        print("\n[2] Fetching Key Metrics...")
        metrics = await client.get_key_metrics(ticker)
        if metrics:
            print(f"Success: PE: {metrics.pe_ratio} | EPS: {metrics.eps}")
        else:
            print("Failed to fetch metrics.")

        # 3. Test News
        print("\n[3] Fetching News...")
        news = await client.get_news(ticker, limit=2)
        for idx, item in enumerate(news, 1):
            print(f"  {idx}. {item.title} ({item.date})")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
