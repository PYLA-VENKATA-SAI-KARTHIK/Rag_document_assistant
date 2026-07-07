# ============================================================
# vector_store.py
# Purpose : Store embeddings in FAISS, search them
# Author  : Venkata Sai Karthik Pyla
# ============================================================

import numpy as np
import faiss
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pdf_loader import load_pdf, chunk_text
from embeddings import generate_embeddings


def build_index(embeddings: list):
    
    # Convert to numpy array (FAISS needs this format)
    vectors = np.array(embeddings).astype('float32')
    
    # Get dimension size (384 in our case)
    dimension = vectors.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Add our vectors to the index
    index.add(vectors)
    
    print(f"✅ Index built! {index.ntotal} vectors stored")
    return index


def search(query: str, index, chunks: list, top_k: int = 3):
    
    print(f"\n🔍 Searching for: '{query}'")
    
    # Convert query to embedding
    query_embedding = generate_embeddings([query])
    
    # Convert to numpy array
    query_vector = np.array(query_embedding).astype('float32')
    
    # Search FAISS index
    distances, indices = index.search(query_vector, top_k)
    
    print(f"✅ Found top {top_k} matches!\n")
    
    # Return matching chunks
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "chunk": chunks[idx],
            "distance": distances[0][i]
        })
    
    return results


if __name__ == "__main__":

    # Step 1: Load PDF
    raw_text = load_pdf("data/resume.pdf")

    # Step 2: Chunk it
    chunks = chunk_text(raw_text)

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    # Step 4: Build FAISS index
    index = build_index(embeddings)

    # Step 5: Search!
    query = "What are Karthik's technical skills?"
    results = search(query, index, chunks)

    # Step 6: Show results
    print("=" * 50)
    print("🎯 SEARCH RESULTS")
    print("=" * 50)

    for result in results:
        print(f"\n📌 Rank {result['rank']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Chunk: {result['chunk'][:200]}...")