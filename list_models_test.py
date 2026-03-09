import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
client = OpenAI(api_key=api_key, base_url=base_url)

print("Listing available Groq models...\n")

try:
    models = client.models.list()
    for model in models.data:
        print(f"- {model.id}")
except Exception as exc:  # noqa: BLE001
    print("Error listing models:", exc)
