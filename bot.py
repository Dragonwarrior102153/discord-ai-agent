import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if DISCORD_TOKEN is None:
    raise ValueError("DISCORD_TOKEN is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Say hello."
)

print(interaction.output_text)