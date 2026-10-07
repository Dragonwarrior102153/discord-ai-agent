"""Load agent settings, workspace location, and authorized Discord user IDs.

Environment values are loaded from this package's ``.env`` file when present.
The allowed-user list is parsed from the comma-separated
``DISCORD_ALLOWED_USER_IDS`` setting.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite"
SYSTEM_INSTRUCTION = (
    "When responding in Discord, do not use LaTeX or LaTeX delimiters "
    "such as $...$ or \\(...\\). Use plain-text mathematical notation "
    "that Discord can display correctly. Use × instead of \\times, "
    "√ instead of \\sqrt{}, and superscript Unicode characters when appropriate. "
    "When using a logarithm, assume base 10 unless another base is specified. "
    "When the input contains user-pinned context, it is selected context from "
    "the current Discord channel; use it only when relevant. A labeled "
    "conversation transcript contains recent messages from this channel in "
    "chronological order. Use relevant earlier messages and respond to the "
    "latest user message."
)

AGENT_WORKSPACE_ROOT = PROJECT_DIR.parent / "agent-workspace"
PYTHON_TOOL_TIMEOUT_SECONDS = 30
AGENT_MAX_TOOL_ITERATIONS = 5

DISCORD_ALLOWED_USER_IDS = frozenset(
    int(user_id.strip())
    for user_id in os.getenv("DISCORD_ALLOWED_USER_IDS", "").split(",")
    if user_id.strip()
)