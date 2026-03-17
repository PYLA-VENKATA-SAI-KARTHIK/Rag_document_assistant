# ============================================================
# embeddings.py
# Purpose : Convert text chunks into number vectors
# Author  : Venkata Sai Karthik Pyla
# ============================================================

import os
import sys

# Tell Python where to find our other files
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sentence_transformers import SentenceTransformer
from pdf_loader import load_pdf, chunk_text

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(chunks):
    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks)
    print(f"✅ Done! Each chunk = {len(embeddings[0])} numbers")
    return embeddings.tolist()


if __name__ == "__main__":

    # Step 1: Load PDF
    raw_text = load_pdf("data/resume.pdf")

    # Step 2: Chunk it
    chunks = chunk_text(raw_text)

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Step 4: Preview
    print("\n" + "="*50)
    print("EMBEDDINGS PREVIEW")
    print("="*50)
    print(f"\nTotal chunks    : {len(embeddings)}")
    print(f"Numbers per chunk: {len(embeddings[0])}")
    print(f"\nFirst chunk text:")
    print(chunks[0])
    print(f"\nFirst 5 numbers:")
    print(embeddings[0][:5])