# ============================================================
# pdf_loader.py
# Purpose : Read PDF and split into chunks
# Author  : Venkata Sai Karthik Pyla
# ============================================================
import re
from pypdf import PdfReader
# New way - works perfectly
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> str:
    print(f"📖 Reading PDF: {file_path}")
    reader = PdfReader(file_path)
    all_text = ""
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            # Fix letter-by-letter spacing issue
            # This detects pattern: "V E N K A T A" 
            # and converts it to: "VENKATA"
            text = re.sub(r'(?<=[A-Za-z]) (?=[A-Za-z])', '', text)
            # Clean extra whitespace
            text = " ".join(text.split())
            all_text += text + " "
    
    print(f"✅ Done! Got {len(all_text)} characters")
    return all_text



def chunk_text(text: str) -> list:
    print("\n✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    raw_text = load_pdf("data/resume.pdf")
    chunks = chunk_text(raw_text)
    print(f"\nFirst chunk:\n{chunks[0]}")