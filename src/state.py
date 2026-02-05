import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    The state of the agent graph.
    It holds the conversation history and the target ticker.
    """

    # 'messages' is a list of chat messages.
    # Annotated[..., operator.add] means new messages are appended to the list, not overwritten.
    messages: Annotated[List[BaseMessage], operator.add]
    ticker: str
