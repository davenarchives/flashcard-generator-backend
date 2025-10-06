from __future__ import annotations

import asyncio
import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import PyPDF2

# -----------------------------------------------
# Load API key from .env file (DO NOT hardcode)
# -----------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Make sure it's in your .env file.")

# Configure Gemini
MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=api_key)

# Tunable knobs for large uploads / concurrency
CHUNK_CHAR_LIMIT = 5500
MAX_CONCURRENT_REQUESTS = 3

# -----------------------------------------------
# Initialize FastAPI app
# -----------------------------------------------
app = FastAPI(title="AI Flashcard Generator")

# Allow frontend requests from the Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://pdflashgen.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------
# Helpers
# -----------------------------------------------

def extract_text(file: UploadFile) -> str:
    """Extract text from PDF or plain text uploads."""
    file.file.seek(0)

    if file.filename.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        text_chunks: List[str] = []
        for page in reader.pages:
            content = page.extract_text() or ""
            if content:
                text_chunks.append(content)
        return "\n".join(text_chunks).strip()

    raw_bytes = file.file.read()
    return raw_bytes.decode("utf-8", errors="ignore").strip()


def chunk_text(text: str, limit: int = CHUNK_CHAR_LIMIT) -> List[str]:
    """Split long documents into reasonably sized chunks."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_len = len(paragraph)
        projected_len = current_len + paragraph_len + (2 if current else 0)

        if projected_len <= limit:
            current.append(paragraph)
            current_len = projected_len
            continue

        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        if paragraph_len > limit:
            for start in range(0, paragraph_len, limit):
                chunks.append(paragraph[start : start + limit])
        else:
            current = [paragraph]
            current_len = paragraph_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_prompt(notes: str, part: int, total_parts: int) -> str:
    part_label = f" (part {part}/{total_parts})" if total_parts > 1 else ""
    return (
        "You are an AI tutor. Convert the following lecture notes" + part_label + " into clear and concise\n"
        "flashcards. Each flashcard must follow this format exactly and avoid numbering:\n\n"
        "Q: <question>\n"
        "A: <answer>\n\n"
        "Notes:\n"
        f"{notes}\n"
    )


async def generate_flashcards(text: str) -> str:
    """Chunk large documents and fan out Gemini requests with a concurrency cap."""
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_chunk(index: int, chunk: str) -> tuple[int, str]:
        prompt = build_prompt(chunk, index + 1, total_chunks)

        def _call_model() -> str:
            model = genai.GenerativeModel(model_name=MODEL_NAME)
            response = model.generate_content(prompt)
            return (response.text or "").strip()

        async with semaphore:
            result = await loop.run_in_executor(None, _call_model)
            return index, result

    if total_chunks == 1:
        # Fast path: avoid scheduling overhead for small uploads
        _, single_result = await process_chunk(0, chunks[0])
        return single_result

    tasks = [process_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    ordered = sorted(results, key=lambda item: item[0])
    combined = "\n\n".join(result for _, result in ordered if result)
    return combined.strip()


# -----------------------------------------------
# Routes
# -----------------------------------------------
@app.get("/")
def home():
    """Simple health check route."""
    return {"message": "AI Flashcard Generator backend is running!"}


@app.post("/summarize")
async def summarize(file: UploadFile):
    """Generate study flashcards using Gemini."""
    text = await run_in_threadpool(extract_text, file)

    if not text:
        return {"error": "No readable text found in uploaded file."}

    try:
        flashcards = await generate_flashcards(text)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    if not flashcards:
        return {"error": "No response generated by Gemini."}

    return {"flashcards": flashcards}
