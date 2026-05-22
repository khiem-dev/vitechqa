from src.loader import load_pdf
from src.chunker import chunk_text
from src.vector_store import create_collection, add_chunks, retrieve

def build_index():
    """
    Chạy 1 lần để build index từ PDF
    """
    print("=== BƯỚC 1: Load PDF ===")
    text = load_pdf("data/CV.pdf")
    
    print("\n=== BƯỚC 2: Chunking ===")
    chunks = chunk_text(text)
    
    print("\n=== BƯỚC 3: Embedding + Lưu vào ChromaDB ===")
    collection = create_collection()
    add_chunks(collection, chunks)
    
    print("\n✅ Build index xong!")
    return collection


def test_retrieve(collection):
    """
    Test thử retrieve với 1 câu hỏi
    """
    query = "GPA bao nhiêu?"
    print(f"\n🔍 Test query: {query}")
    
    chunks = retrieve(collection, query, top_k=3)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Kết quả {i+1} ---")
        print(chunk[:300])


if __name__ == "__main__":
    collection = build_index()
    test_retrieve(collection)