import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

print("Sending request...")
start = time.time()

response = client.interactions.create(
    model="gemini-3.8-flash",
    input="Say hello."
)

end = time.time()

print(f"Response received in {end - start:.2f} seconds")
print(response.output_text)