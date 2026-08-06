# ============================================================
# llm_handler.py
# Purpose : Connect Groq LLM to answer questions
# Author  : Venkata Sai Karthik Pyla
# ============================================================

import os
import sys
import warnings
import logging
from dotenv import load_dotenv
from groq import Groq, AuthenticationError

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load API keys from .env
load_dotenv()

# Setup Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if groq_api_key else None


def build_prompt(question: str, context_chunks: list) -> str:
    """
    Combine question + relevant chunks into one prompt
    This is PROMPT ENGINEERING!
    """

    context = "\n\n".join([
        f"Context {i+1}: {chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    prompt = f"""You are a helpful assistant that answers 
questions based ONLY on the provided context below.

If the answer is not in the context, say:
"I could not find that information in the document."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    return prompt


def get_answer(question: str, context_chunks: list) -> str:
    """
    Send question + context to Groq → get answer!
    """

    print(f"\n🤖 Sending to Groq...")

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add a valid Groq API key to your .env file."
        )

    prompt = build_prompt(question, context_chunks)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based only on provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
    except AuthenticationError as exc:
        raise RuntimeError(
            "Groq rejected the API key. Check that GROQ_API_KEY in your .env file is valid."
        ) from exc

    print(f"✅ Got answer from Groq!")
    return response.choices[0].message.content or "I could not generate a response."


if __name__ == "__main__":

    from pdf_loader import load_pdf, chunk_text
    from embeddings import generate_embeddings
    from vector_store import build_index, search

    # Step 1: Load PDF
    raw_text = load_pdf("data/resume.pdf")
    chunks = chunk_text(raw_text)

    # Step 2: Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Step 3: Build FAISS index
    index = build_index(embeddings)

    # Step 4: Test multiple questions!
    questions = [
        "What are Karthik's technical skills?",
        "What projects has Karthik built?",
        "What is Karthik's CGPA?"
    ]

    for question in questions:
        # Search relevant chunks
        results = search(question, index, chunks)

        # Extract chunk text
        context_chunks = [r["chunk"] for r in results]

        # Get answer from Groq
        answer = get_answer(question, context_chunks)

        print("\n" + "="*50)
        print(f"❓ Question: {question}")
        print(f"\n🤖 Answer:\n{answer}")
        print("="*50)