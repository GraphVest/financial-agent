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

    # Track how many messages we've already logged to avoid duplicates
    logged_count = 1  # We already logged the initial HumanMessage

    # 3. Run Graph
    async for event in app.astream(initial_state, stream_mode="values"):
        messages = event.get("messages")
        if messages:
            # Log ALL new messages, not just the last one
            new_messages = messages[logged_count:]
            
            for msg in new_messages:
                if isinstance(msg, HumanMessage):
                    continue
                    
                logger.log(msg)
                
                print(f"✅ Processed step: {type(msg).__name__}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"   -> Calling {len(msg.tool_calls)} tool(s)...")
            
            # Update counter to current message count
            logged_count = len(messages)

    print(f"\n🎉 Research Complete! Check the 'logs/' folder for the report: {logger.md_filename}")


if __name__ == "__main__":
    asyncio.run(main())
