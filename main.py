import asyncio

from langchain_core.messages import HumanMessage

from src.graph import app
from src.utils import MarkdownLogger


async def main():
    ticker = "NVDA"
    print(f"🚀 Starting Financial Research Agent for: {ticker}...\n")

    # 1. Init Logger
    logger = MarkdownLogger(ticker)

    # 2. Init State
    initial_state = {"messages": [HumanMessage(content=f"Research {ticker} stock.")], "ticker": ticker}

    logger.log(initial_state["messages"][0])

    # 3. Run Graph
    async for event in app.astream(initial_state, stream_mode="values"):
        messages = event.get("messages")
        if messages:
            last_message = messages[-1]

            if isinstance(last_message, HumanMessage):
                continue

            logger.log(last_message)

            print(f"✅ Processed step: {type(last_message).__name__}")
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"   -> Calling {len(last_message.tool_calls)} tool(s)...")

    # --- SỬA DÒNG NÀY ---
    # Cũ (Lỗi): logger.filename
    # Mới (Đúng): logger.md_filename
    print(f"\n🎉 Research Complete! Check the 'logs/' folder for the report: {logger.md_filename}")


if __name__ == "__main__":
    asyncio.run(main())
