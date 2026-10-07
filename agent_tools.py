"""
Workspace tools and the Gemini function-calling loop used by the Discord agent.

File operations are confined to AGENT_WORKSPACE_ROOT. Python code runs in a
subprocess with the bot process's operating-system permissions; it is not an
operating-system security sandbox.
"""

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from google.genai import errors

from config import (
    AGENT_WORKSPACE_ROOT,
    PYTHON_TOOL_TIMEOUT_SECONDS,
    AGENT_MAX_TOOL_ITERATIONS,
    GEMINI_FALLBACK_MODEL,
    GEMINI_MODEL,
    SYSTEM_INSTRUCTION,
)

AGENT_WORKSPACE_ROOT.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "name": "list_files",
        "description": "List files and directories in the agent workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": 'Directory relative to the workspace root. Defaults to ".".',
                }
            },
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a UTF-8 text file from the agent workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "File path relative to the workspace root.",
                }
            },
            "required": ["filepath"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Write UTF-8 text to a file in the agent workspace, creating parent directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "File path relative to the workspace root.",
                },
                "content": {"type": "string", "description": "Text to write."},
            },
            "required": ["filepath", "content"],
        },
    },
    {
        "type": "function",
        "name": "run_python",
        "description": "Run Python code in the agent workspace. This code has the bot's OS permissions.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (maximum 30).",
                },
            },
            "required": ["code"],
        },
    },
]


def _validate_path(path: str | Path) -> Path:
    """Ensure path is within AGENT_WORKSPACE_ROOT. Raises ValueError if escape attempted."""
    workspace_root = AGENT_WORKSPACE_ROOT.resolve()
    resolved = (workspace_root / path).resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        raise ValueError(f"Path escape attempt: {path}")
    return resolved


def list_files(directory: str = ".") -> list[str]:
    """List all files and directories in the agent workspace (relative to root)."""
    base_dir = _validate_path(directory)
    if not base_dir.exists():
        return []
    
    items = []
    for item in sorted(base_dir.iterdir()):
        rel_path = item.relative_to(AGENT_WORKSPACE_ROOT)
        if item.is_dir():
            items.append(f"{rel_path}/")
        else:
            items.append(str(rel_path))
    return items


def read_file(filepath: str) -> str:
    """Read file contents from the agent workspace."""
    file_path = _validate_path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not file_path.is_file():
        raise ValueError(f"Not a file: {filepath}")
    return file_path.read_text(encoding="utf-8")


def write_file(filepath: str, content: str) -> str:
    """Write content to a file in the agent workspace. Creates parent directories as needed."""
    file_path = _validate_path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return f"Written {len(content)} chars to {filepath}"


def run_python(
    code: str, timeout: int = PYTHON_TOOL_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """
    Execute Python code in the agent workspace context.
    Returns dict with 'success', 'output', 'error', and 'returncode'.
    """
    if not 1 <= timeout <= PYTHON_TOOL_TIMEOUT_SECONDS:
        return {
            "success": False,
            "output": "",
            "error": f"timeout must be between 1 and {PYTHON_TOOL_TIMEOUT_SECONDS} seconds",
            "returncode": -1,
        }

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(AGENT_WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Execution timeout (>{timeout}s)",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "returncode": -1,
        }


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_python": run_python,
}


@dataclass(frozen=True)
class ToolExecutionStatus:
    """Record whether one agent tool invocation succeeded."""

    name: str
    succeeded: bool


@dataclass(frozen=True)
class AgentRunResult:
    """Collect the final response, tool outcomes, and completion state."""

    response_text: str
    tool_executions: tuple[ToolExecutionStatus, ...]
    completed: bool = True


def execute_tool(name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
    """Run a declared tool and return its result and whether it failed."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}, True

    try:
        result = function(**arguments)
    except (OSError, ValueError, TypeError) as error:
        return {"error": str(error)}, True

    if isinstance(result, dict) and result.get("success") is False:
        return result, True
    return result, False


def _create_interaction(
    client: Any, *, model: str, **kwargs: Any
) -> tuple[Any, str]:
    """Create an interaction and retry once on the configured fallback after 429."""
    try:
        return client.interactions.create(model=model, **kwargs), model
    except errors.APIError as error:
        if error.code != 429 or model == GEMINI_FALLBACK_MODEL:
            raise

        logger.warning(
            "Gemini rate limit received for model %s; retrying with %s",
            model,
            GEMINI_FALLBACK_MODEL,
        )
        return (
            client.interactions.create(model=GEMINI_FALLBACK_MODEL, **kwargs),
            GEMINI_FALLBACK_MODEL,
        )


def _label_model_response(text: str, model: str) -> str:
    """Mark responses generated with the secondary model."""
    if model == GEMINI_FALLBACK_MODEL:
        return f"[2] {text}"
    return text


def run_agent(client: Any, prompt: str, *, enable_tools: bool) -> AgentRunResult:
    """Run a Gemini interaction, fulfilling tool calls when the caller is authorized."""
    tools = TOOL_DECLARATIONS if enable_tools else None
    tool_executions: list[ToolExecutionStatus] = []
    active_model = GEMINI_MODEL
    response, active_model = _create_interaction(
        client,
        model=active_model,
        system_instruction=SYSTEM_INSTRUCTION,
        input=prompt,
        tools=tools,
    )

    for _ in range(AGENT_MAX_TOOL_ITERATIONS):
        function_calls = [
            step
            for step in (response.steps or [])
            if step.type == "function_call"
        ]
        if not function_calls:
            response_text = response.output_text or "Gemini returned no text response."
            return AgentRunResult(
                response_text=_label_model_response(response_text, active_model),
                tool_executions=tuple(tool_executions),
            )
        if not enable_tools:
            raise RuntimeError("Gemini requested a tool call for an unauthorized user.")
        if not response.id:
            raise RuntimeError("Gemini tool-call response did not include an interaction ID.")

        function_results = []
        for call in function_calls:
            result, is_error = execute_tool(call.name, call.arguments)
            tool_executions.append(
                ToolExecutionStatus(name=call.name, succeeded=not is_error)
            )
            function_results.append(
                {
                    "type": "function_result",
                    "call_id": call.id,
                    "name": call.name,
                    "result": json.dumps(result, ensure_ascii=False),
                    "is_error": is_error,
                }
            )

        response, active_model = _create_interaction(
            client,
            model=active_model,
            previous_interaction_id=response.id,
            input=function_results,
        )

    response_text = (
        response.output_text
        or "I reached the tool-call limit before finishing. Please ask me to continue."
    )
    return AgentRunResult(
        response_text=_label_model_response(response_text, active_model),
        tool_executions=tuple(tool_executions),
        completed=False,
    )


def format_completion_status(
    executions: tuple[ToolExecutionStatus, ...], *, completed: bool = True
) -> str:
    """Format a short, deterministic status footer for the user."""
    successful = list(dict.fromkeys(
        execution.name for execution in executions if execution.succeeded
    ))
    failed = list(dict.fromkeys(
        execution.name for execution in executions if not execution.succeeded
    ))
    parts = []
    if not completed:
        parts.append("Error: agent reached its tool-call limit before completing.")
    if successful:
        parts.append(f"Finished: {', '.join(successful)}.")
    if failed:
        parts.append(f"Error: {', '.join(failed)} failed.")
    if not parts:
        return "✅ Finished."
    return ("⚠️ " if failed or not completed else "✅ ") + " ".join(parts)


if __name__ == "__main__":
    print(f"Agent workspace: {AGENT_WORKSPACE_ROOT}")
    print(f"Files: {list_files()}")
