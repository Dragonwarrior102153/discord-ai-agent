# Discord client that routes messages from any channel to Gemini.

import asyncio
import logging
import os
import sqlite3

import discord
from discord.ext import tasks
from google import genai

from agent_tools import format_completion_status, run_agent
from config import DISCORD_ALLOWED_USER_IDS
from memory import (
    clear_history,
    delete_expired_histories,
    format_history_for_prompt,
    get_history,
    get_pinned_messages,
    pin_message,
    save_message,
    unpin_message,
)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is missing; set it in discord-ai-agent/.env.")

gemini = genai.Client()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True


class AgentBot(discord.Client):
    # Discord client with application-command registration.

    def __init__(self, *, intents: discord.Intents) -> None:
        super().__init__(intents=intents)
        self.tree = discord.app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # Register slash commands once during client startup.
        try:
            await self.tree.sync()
        except discord.HTTPException:
            logger.exception("Failed to register Discord application commands")


bot = AgentBot(intents=intents)


@tasks.loop(hours=1)
async def expire_inactive_memory() -> None:
    try:
        await asyncio.to_thread(delete_expired_histories)
    except (OSError, sqlite3.Error):
        logger.exception("Failed to expire inactive conversation histories")


@bot.tree.command(name="clear", description="Clear conversation memory for this channel")
async def clear_channel_memory(interaction: discord.Interaction) -> None:
    # Clear the conversation history and pinned context for this channel ID.
    try:
        await asyncio.to_thread(clear_history, interaction.channel_id)
    except (OSError, sqlite3.Error):
        logger.exception(
            "Failed to clear conversation memory for channel %s",
            interaction.channel_id,
        )
        await interaction.response.send_message(
            "❌ I couldn't clear this channel's conversation memory because of a database error.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        "✅ Conversation history and pinned context for this channel have been cleared.",
        ephemeral=True,
    )


@bot.tree.command(name="pin", description="Pin important context for this channel")
@discord.app_commands.describe(
    content="Important context to keep in this channel (up to 500 characters)"
)
async def pin_channel_context(
    interaction: discord.Interaction, content: str
) -> None:
    try:
        pin_id = await asyncio.to_thread(pin_message, interaction.channel_id, content)
    except ValueError as error:
        await interaction.response.send_message(f"⚠️ {error}", ephemeral=True)
        return
    except (OSError, sqlite3.Error):
        logger.exception(
            "Failed to pin context for channel %s",
            interaction.channel_id,
        )
        await interaction.response.send_message(
            "❌ I couldn't pin that context because of a database error.",
            ephemeral=True,
        )
        return

    await interaction.response.send_message(
        f"✅ Pinned channel context as #{pin_id}. Use `/pins` to list pins or `/unpin` to remove one.",
        ephemeral=True,
    )


@bot.tree.command(name="pins", description="List pinned context for this channel")
async def list_channel_pins(interaction: discord.Interaction) -> None:
    try:
        pins = await asyncio.to_thread(get_pinned_messages, interaction.channel_id)
    except (OSError, sqlite3.Error):
        logger.exception(
            "Failed to list pinned context for channel %s",
            interaction.channel_id,
        )
        await interaction.response.send_message(
            "❌ I couldn't list pinned context because of a database error.",
            ephemeral=True,
        )
        return

    if not pins:
        await interaction.response.send_message(
            "There is no pinned context in this channel.",
            ephemeral=True,
        )
        return

    pin_list = "\n".join(f"#{pin.id}: {pin.content}" for pin in pins)
    await interaction.response.send_message(pin_list, ephemeral=True)


@bot.tree.command(name="unpin", description="Remove a pinned context entry from this channel")
@discord.app_commands.describe(pin_id="Pin ID shown by `/pins`")
async def remove_channel_pin(
    interaction: discord.Interaction, pin_id: int
) -> None:
    try:
        removed = await asyncio.to_thread(
            unpin_message, interaction.channel_id, pin_id
        )
    except (OSError, sqlite3.Error):
        logger.exception(
            "Failed to remove pinned context for channel %s",
            interaction.channel_id,
        )
        await interaction.response.send_message(
            "❌ I couldn't remove that pin because of a database error.",
            ephemeral=True,
        )
        return

    confirmation = (
        f"✅ Removed pin #{pin_id} from this channel."
        if removed
        else f"⚠️ Pin #{pin_id} was not found in this channel."
    )
    await interaction.response.send_message(confirmation, ephemeral=True)


@bot.event
async def on_ready():
    # Log the Discord account after the client connects.
    try:
        await asyncio.to_thread(delete_expired_histories)
    except (OSError, sqlite3.Error):
        logger.exception("Failed to initialize or clean conversation memory")
    if not expire_inactive_memory.is_running():
        expire_inactive_memory.start()
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message):
    # Process channel messages and reply with the agent's result.
    if message.author.bot:
        return

    can_use_tools = message.author.id in DISCORD_ALLOWED_USER_IDS

    try:
        prompt = message.content
        memory_available = False
        try:
            await asyncio.to_thread(save_message, message.channel.id, "user", message.content)
            history = await asyncio.to_thread(get_history, message.channel.id)
            pinned_messages = await asyncio.to_thread(
                get_pinned_messages, message.channel.id
            )
            if history:
                prompt = format_history_for_prompt(history, pinned_messages)
                memory_available = True
            else:
                logger.warning(
                    "Conversation history is empty for channel %s; using current message only",
                    message.channel.id,
                )
        except (OSError, sqlite3.Error):
            logger.exception(
                "Failed to load conversation memory for channel %s; using current message only",
                message.channel.id,
            )

        result = await asyncio.to_thread(
            run_agent,
            gemini,
            prompt,
            enable_tools=can_use_tools,
        )
        if memory_available:
            try:
                await asyncio.to_thread(
                    save_message, message.channel.id, "model", result.response_text
                )
            except (OSError, sqlite3.Error):
                logger.exception(
                    "Failed to save Gemini response for channel %s",
                    message.channel.id,
                )

        response_text = (
            f"{result.response_text}\n\n"
            f"{format_completion_status(result.tool_executions, completed=result.completed)}"
        )
        chunks = [
            response_text[start : start + 2000]
            for start in range(0, len(response_text), 2000)
        ]
        await message.reply(chunks[0], mention_author=False)
        for chunk in chunks[1:]:
            await message.channel.send(chunk)
    except Exception:
        logger.exception("Failed to process a Discord message")
        await message.reply(
            "❌ Error: I couldn't complete that command because an internal error occurred.",
            mention_author=False,
        )


bot.run(DISCORD_TOKEN)