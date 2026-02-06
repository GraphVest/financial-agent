import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class MarkdownLogger:
    """
    A Dual-Logger that streams agent interactions to a readable Markdown file
    while simultaneously archiving the full raw context into a structured JSON file.

    - Markdown (.md): For human readability - contains final report + trace (no raw data)
    - JSON (.json): For machine processing - contains metadata, messages, and extracted data
    """

    def __init__(self, ticker: str, log_dir: Path | None = None, save_interval: int = 1):
        """
        Initialize the dual logger.

        Args:
            ticker: Stock ticker symbol
            log_dir: Directory to save logs (default: logs/ relative to source file)
            save_interval: Save JSON every N messages (default: 1 = save every message)
        """
        # Capture timestamp once for consistency across all uses
        now = datetime.now()
        self.start_time = now.isoformat()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.ticker = ticker

        # Batch save configuration
        self.save_interval = max(1, save_interval)  # Ensure at least 1
        self._message_count = 0

        # Use provided log_dir or default to 'logs' relative to this source file
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        self.log_dir = Path(log_dir)

        # Ensure the logs directory exists
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create log directory {self.log_dir}: {e}")
            raise

        # Initialize filenames
        self.base_filename = self.log_dir / f"{ticker}_RESEARCH_{timestamp}"
        self.md_filename = self.base_filename.with_suffix(".md")
        self.json_filename = self.base_filename.with_suffix(".json")

        # Initialize the Markdown file with a header
        try:
            with open(self.md_filename, "w", encoding="utf-8") as f:
                f.write(f"# 🕵️‍♂️ Financial Research Log: ${ticker}\n")
                f.write(f"*Date: {now.strftime('%Y-%m-%d %H:%M:%S')}*\n")
                f.write(
                    f"*Raw Data Context:* [`{self.json_filename.name}`]"
                    f"(./{self.json_filename.name})\n\n"
                )
                f.write("---\n\n")
        except IOError as e:
            logging.error(f"Failed to initialize Markdown log file {self.md_filename}: {e}")
            raise

        # Initialize the structured JSON data
        self.json_data: dict[str, Any] = {
            "metadata": {
                "ticker": ticker,
                "timestamp": self.start_time,
                "tools_called": [],
            },
            "raw_messages": [],
            "extracted_data": {
                "profile": None,
                "metrics": None,
                "news": [],
                "income_statement": [],
                "balance_sheet": [],
                "cash_flow": [],
            },
        }

        print(
            f"📝 Dual Logger initialized:\n   - Human Log: {self.md_filename}\n   - Machine Data: {self.json_filename}"
        )

    def flush(self):
        """Force save JSON data to disk. Call this at end of session to ensure all data is saved."""
        self._save_json()

    def _extract_tool_data(self, tool_name: str, content: Any):
        """Extract and store tool output data in the structured extracted_data section."""
        if not isinstance(content, (dict, list)):
            # Try to parse if it's a JSON string
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (ValueError, TypeError):
                    return

        # Map tool outputs to extracted_data fields
        if tool_name == "get_company_profile":
            self.json_data["extracted_data"]["profile"] = content
        elif tool_name == "get_financial_ratios":
            self.json_data["extracted_data"]["metrics"] = content
        elif tool_name == "get_stock_news":
            self.json_data["extracted_data"]["news"] = content if isinstance(content, list) else [content]
        elif tool_name == "get_financial_statements":
            if isinstance(content, dict):
                if "income_statement" in content:
                    self.json_data["extracted_data"]["income_statement"] = content["income_statement"]
                if "balance_sheet" in content:
                    self.json_data["extracted_data"]["balance_sheet"] = content["balance_sheet"]
                if "cash_flow" in content:
                    self.json_data["extracted_data"]["cash_flow"] = content["cash_flow"]

    def _save_json(self):
        """Save the structured JSON data to disk."""
        with open(self.json_filename, "w", encoding="utf-8") as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)

    def log(self, message: BaseMessage):
        """
        Processes a LangChain message:
        1. Appends the raw message data to the JSON context file.
        2. Formats and writes the message to the Markdown log file.

        For ToolMessages, we store only a lightweight reference in raw_messages
        to avoid data duplication. The actual tool output data is stored in
        extracted_data for structured access.
        """

        # --- PART A: JSON Archiving (Machine Readable) ---

        # Parse ToolMessage content if it's a JSON string
        parsed_content = message.content
        if isinstance(message, ToolMessage):
            try:
                parsed_content = json.loads(message.content)
            except (json.JSONDecodeError, TypeError):
                pass  # Keep as string if parsing fails

        # Construct the log entry for raw_messages
        entry = {
            "type": type(message).__name__,
            "timestamp": datetime.now().isoformat(),
        }

        # For ToolMessages: store only a reference, not the full content (avoid duplication)
        # The actual data is stored in extracted_data for structured access
        if isinstance(message, ToolMessage):
            tool_name = self._get_tool_name_from_id(message.tool_call_id)
            entry["tool_call_id"] = message.tool_call_id
            entry["tool_name"] = tool_name or "unknown"
            entry["content_ref"] = self._get_extracted_data_path(tool_name)

            # Extract and store data in structured extracted_data section
            if tool_name:
                self._extract_tool_data(tool_name, parsed_content)
        else:
            # For non-ToolMessages, store the full content
            entry["content"] = parsed_content

        # Capture tool calls if present (for AI reasoning steps)
        if hasattr(message, "tool_calls") and message.tool_calls:
            entry["tool_calls"] = message.tool_calls
            # Track tools called in metadata
            for tool in message.tool_calls:
                tool_name = tool.get("name", "")
                if tool_name and tool_name not in self.json_data["metadata"]["tools_called"]:
                    self.json_data["metadata"]["tools_called"].append(tool_name)

        # Update raw_messages
        self.json_data["raw_messages"].append(entry)

        # Save JSON to disk (batch save based on save_interval)
        self._message_count += 1
        if self._message_count >= self.save_interval:
            self._save_json()
            self._message_count = 0

        # --- PART B: Markdown Logging (Human Readable) ---

        try:
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

                # 3. Tool Message - Only reference, no raw data
                elif isinstance(message, ToolMessage):
                    tool_name = self._get_tool_name_from_id(message.tool_call_id)
                    f.write(f"### 📬 Tool Output: `{tool_name or message.tool_call_id}`\n")
                    f.write(
                        f"> ✅ Data received. Full data: [{self.json_filename.name}]"
                        f"(./{self.json_filename.name})\n\n"
                    )

                # 4. System Message
                elif isinstance(message, SystemMessage):
                    content = message.content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    f.write(f"**System Context:** *{content}*\n\n")

                # 5. Unknown message type
                else:
                    logging.warning(f"Unknown message type: {type(message).__name__}")
                    f.write(f"## ❓ Unknown Message ({type(message).__name__})\n")
                    f.write(f"{message.content}\n\n")

        except IOError as e:
            logging.error(f"Failed to write to Markdown log file {self.md_filename}: {e}")

    def _get_extracted_data_path(self, tool_name: str) -> str:
        """
        Returns a reference path to where the tool data is stored in extracted_data.
        This is used in raw_messages to point to the actual data without duplication.
        """
        path_map = {
            "get_company_profile": "extracted_data.profile",
            "get_financial_ratios": "extracted_data.metrics",
            "get_stock_news": "extracted_data.news",
            "get_financial_statements": "extracted_data.{income_statement, balance_sheet, cash_flow}",
        }
        return path_map.get(tool_name, f"extracted_data.{tool_name}")

    def _get_tool_name_from_id(self, tool_call_id: str) -> str:
        """Find the tool name from a tool_call_id by searching previous messages."""
        for msg in self.json_data["raw_messages"]:
            if "tool_calls" in msg:
                for tool in msg["tool_calls"]:
                    if tool.get("id") == tool_call_id:
                        return tool.get("name", "")
        return ""

