"""Send a minimal Gemini request and print its response time and text.

This standalone smoke test requires ``GEMINI_API_KEY`` in the environment or
in the package's ``.env`` file.
"""

import os
import time

from google import genai

from config import GEMINI_MODEL

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file or environment.")

client = genai.Client(api_key=api_key)

print("Sending request...")
start = time.time()

chat = client.chats.create(model=GEMINI_MODEL)
response = chat.send_message("Say hello.")

end = time.time()
print(f"Response received in {end - start:.2f} seconds")
print(response.text)
