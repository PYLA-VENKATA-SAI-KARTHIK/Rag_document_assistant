# ============================================================
# rag_pipeline.py
# Purpose : Complete RAG pipeline in one place!
# Author  : Venkata Sai Karthik Pyla
# ============================================================

import os
import sys
import warnings
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pdf_loader import load_pdf, chunk_text
from embeddings import generate_embeddings
from vector_store import build_index, search
from llm_handler import get_answer


class RAGPipeline:
    """
    Complete RAG Pipeline in one class!
    
    This is called a CLASS!
    Think of it like a blueprint for a machine!
    
    The machine:
    1. Loads a PDF once
    2. Answers unlimited questions!
    """
    
    def __init__(self, pdf_path: str):
        """
        This runs ONCE when you create the pipeline!
        Like turning on the machine!
        """
        
        print("\n🚀 Initializing RAG Pipeline...")
        print("="*50)
        
        # Step 1: Load PDF
        self.raw_text = load_pdf(pdf_path)
        
        # Step 2: Chunk text
        self.chunks = chunk_text(self.raw_text)
        
        # Step 3: Generate embeddings
        self.embeddings = generate_embeddings(self.chunks)
        
        # Step 4: Build FAISS index
        self.index = build_index(self.embeddings)
        
        print("="*50)
        print("✅ Pipeline Ready! Ask me anything!\n")
    
    
    def ask(self, question: str, top_k: int = 3) -> str:
        """
        Ask any question → get answer!
        This can be called unlimited times!
        """
        
        # Step 5: Search relevant chunks
        results = search(question, self.index, self.chunks, top_k)
        
        # Step 6: Extract chunk text
        context_chunks = [r["chunk"] for r in results]
        
        # Step 7: Get LLM answer
        answer = get_answer(question, context_chunks)
        
        return answer


def main():
    """
    Main function — runs the interactive Q&A!
    """
    
    # Initialize pipeline with resume
    pipeline = RAGPipeline("data/resume.pdf")
    
    # Interactive loop!
    print("💬 Ask questions about the document!")
    print("   Type 'quit' to exit\n")
    
    while True:
        
        # Get question from user
        question = input("❓ Your question: ").strip()
        
        # Exit condition
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty questions
        if not question:
            print("Please type a question!\n")
            continue
        
        # Get answer
        print("\n⏳ Thinking...")
        answer = pipeline.ask(question)
        
        # Show answer
        print(f"\n🤖 Answer: {answer}")
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()