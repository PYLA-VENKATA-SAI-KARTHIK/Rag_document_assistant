# ============================================================
# pdf_loader.py
# Purpose : Read PDF and split into clean chunks
# Author  : Venkata Sai Karthik Pyla
# ============================================================

import re
import os
import sys
import warnings
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Clean all PDF spacing issues"""

    # Fix letter spacing: "V E N K A T A" → "VENKATA"
    text = re.sub(r'(?<=[A-Za-z]) (?=[A-Za-z])', '', text)

    # Fix number spacing: "2 0 2 7" → "2027"
    text = re.sub(r'(?<=\d) (?=\d)', '', text)

    # Fix punctuation spacing
    text = re.sub(r' \. ', '.', text)
    text = re.sub(r' \- ', '-', text)
    text = re.sub(r' \@ ', '@', text)
    text = re.sub(r' \/ ', '/', text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def load_pdf(file_path: str) -> str:
    """Read PDF and return clean text"""

    print(f"📖 Reading PDF: {file_path}")
    reader = PdfReader(file_path)
    all_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text = clean_text(text)
            all_text += text + " "

    print(f"✅ Done! Got {len(all_text)} characters")
    return all_text


def chunk_text(text: str) -> list:
    """Split text into clean overlapping chunks"""

    print("\n✂️  Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    # Filter noise chunks shorter than 150 characters
    chunks = [c for c in chunks if len(c) > 150]

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    raw_text = load_pdf("data/resume.pdf")
    chunks = chunk_text(raw_text)

    print("\n" + "="*50)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)