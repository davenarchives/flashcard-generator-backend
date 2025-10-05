import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("📋 Listing available Gemini models...\n")

try:
    models = genai.list_models()
    for m in models:
        # only show models that support text generation
        if "generateContent" in m.supported_generation_methods:
            print(f"✅ {m.name} — supports: {m.supported_generation_methods}")
except Exception as e:
    print("❌ Error listing models:", e)
