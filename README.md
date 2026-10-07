# Discord AI Agent

A Discord bot for delegating tasks to a Gemini-powered agent.

Shared model and agent settings live in [`config.py`](./config.py). The Discord
bot and the local example use the same model, instructions, tool declarations,
and agent loop.

Add `DISCORD_ALLOWED_USER_IDS` to `.env` as a comma-separated list of Discord
user IDs allowed to use agent tools, for example
`DISCORD_ALLOWED_USER_IDS=123456789012345678,234567890123456789`. Users not on
this list can still chat with the bot in any channel, but Gemini is not given
the file or Python tools for their requests. Leave it empty to disable tool
access.

Conversation history is stored locally per channel in `data/memory.db`. The bot
keeps at most the latest 20 messages per channel and deletes channel history
after 7 days of inactivity. Use `/pin` to preserve up to three short context
notes per channel; `/pins` lists them, `/unpin` removes one, and `/clear`
deletes the current channel's history and pins.

If the primary Gemini model is rate-limited, the agent retries with
`gemini-2.5-flash-lite`. Replies from that fallback model start with `[2]`.

The file tools use the repository-level `agent-workspace/` directory. Python
execution runs with the bot's operating-system permissions; the workspace path
does not sandbox or restrict code execution. Only authorize users you trust to
run code on the computer hosting the bot.
