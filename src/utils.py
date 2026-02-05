import json
import os
from datetime import datetime
from typing import Any, Dict, List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class MarkdownLogger:
    """
    A Dual-Logger that streams agent interactions to a readable Markdown file
    while simultaneously archiving the full raw context into a JSON file.

    - Markdown (.md): For human readability, debugging, and interviewing.
    - JSON (.json): For machine processing (e.g., loading data for Valuation models).
    """

    def __init__(self, ticker: str):
        # Ensure the logs directory exists
        os.makedirs("logs", exist_ok=True)

        # Create a unique timestamped base filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_filename = f"logs/{ticker}_RESEARCH_{timestamp}"

        # 1. Initialize the Markdown file with a header
        self.md_filename = f"{self.base_filename}.md"
        self.json_filename = f"{self.base_filename}.json"

        with open(self.md_filename, "w", encoding="utf-8") as f:
            f.write(f"# 🕵️‍♂️ Financial Research Log: ${ticker}\n")
            f.write(f"*Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
            f.write(
                f"*Raw Data Context:* [`{os.path.basename(self.json_filename)}`]"
                f"(./{os.path.basename(self.json_filename)})\n\n"
            )
            f.write("---\n\n")

        # 2. Initialize the in-memory buffer for JSON context
        self.context_buffer: List[Dict[str, Any]] = []

        print(
            f"📝 Dual Logger initialized:\n   - Human Log: {self.md_filename}\n   - Machine Data: {self.json_filename}"
        )

    def _format_json(self, data: Union[str, Dict, List]) -> str:
        """Helper to pretty-print JSON objects for the Markdown report."""
        if isinstance(data, str):
            try:
                # Attempt to parse string as JSON to pretty print it
                parsed = json.loads(data)
                return json.dumps(parsed, indent=2)
            except ValueError:
                return data
        return json.dumps(data, indent=2)

    def log(self, message: BaseMessage):
        """
        Processes a LangChain message:
        1. Appends the raw message data to the JSON context file.
        2. Formats and writes the message to the Markdown log file.
        """

        # --- PART A: JSON Archiving (Machine Readable) ---

        # Extract content (handle cases where content is a JSON string)
        msg_content = message.content
        if isinstance(message, ToolMessage):
            try:
                msg_content = json.loads(message.content)
            except Exception:
                pass  # Keep as string if parsing fails

        # Construct the log entry
        entry = {"type": type(message).__name__, "timestamp": datetime.now().isoformat(), "content": msg_content}

        # Capture tool calls if present (for AI reasoning steps)
        if hasattr(message, "tool_calls") and message.tool_calls:
            entry["tool_calls"] = message.tool_calls

        # Capture tool call ID if present (for Tool outputs)
        if hasattr(message, "tool_call_id"):
            entry["tool_call_id"] = message.tool_call_id

        # Update buffer and dump to disk immediately (atomic write mostly)
        self.context_buffer.append(entry)
        with open(self.json_filename, "w", encoding="utf-8") as f:
            json.dump(self.context_buffer, f, indent=2, ensure_ascii=False)

        # --- PART B: Markdown Logging (Human Readable) ---

        with open(self.md_filename, "a", encoding="utf-8") as f:
            # 1. Human Message
            if isinstance(message, HumanMessage):
                f.write("## 👤 User Request\n")
                f.write(f"> {message.content}\n\n")

            # 2. AI Message
            elif isinstance(message, AIMessage):
                # Case: AI is executing tools
                if message.tool_calls:
                    f.write("## 🤖 Agent Reasoning & Actions\n")
                    if message.content:
                        f.write(f"{message.content}\n\n")

                    for tool in message.tool_calls:
                        f.write(f"### 🛠️ Executing Tool: `{tool['name']}`\n")
                        f.write(f"```json\n{json.dumps(tool['args'], indent=2)}\n```\n")

                # Case: AI is providing the final answer
                else:
                    f.write("## 📝 Final Output\n")
                    f.write(f"{message.content}\n\n")

            # 3. Tool Message (The raw data return)
            elif isinstance(message, ToolMessage):
                f.write(f"### 📬 Tool Output (ID: `{message.tool_call_id}`)\n")

                # Use HTML details tag to collapse heavy JSON data
                formatted_content = self._format_json(message.content)
                f.write("<details>\n<summary>Click to view raw data</summary>\n\n")
                f.write(f"```json\n{formatted_content}\n```\n")
                f.write("\n</details>\n\n")

            # 4. System Message
            elif isinstance(message, SystemMessage):
                f.write(f"**System Context:** *{message.content[:100]}...*\n\n")
