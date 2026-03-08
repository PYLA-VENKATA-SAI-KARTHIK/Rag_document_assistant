# ============================================================
# 📄 pdf_loader.py
# Purpose : Read PDF and split into chunks
# Author  : Venkata Sai Karthik Pyla
# ============================================================

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> str:
    """Read a PDF and return all text"""
    
    print(f"📖 Reading PDF: {file_path}")
    
    reader = PdfReader(file_path)
    all_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text += text
    
    print(f"✅ Done! Got {len(all_text)} characters")
    return all_text


def chunk_text(text: str) -> list:
    """Split big text into small chunks"""
    
    print("\n✂️  Splitting into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_text(text)
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


# ============================================================
# Run this file directly to test:
# python src/pdf_loader.py
# ============================================================

if __name__ == "__main__":

    # 👇 ONLY change this one line — just the filename
    PDF_PATH = "data/resume.pdf"

    # Load PDF
    raw_text = load_pdf(PDF_PATH)

    # Chunk it
    chunks = chunk_text(raw_text)

    # Preview
    print("\n" + "="*50)
    print("📦 FIRST 3 CHUNKS PREVIEW")
    print("="*50)

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)
