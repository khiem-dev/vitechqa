import chromadb
from .embedder import embed_chunks, embed_query

# Khởi tạo ChromaDB lưu trên disk
client = chromadb.PersistentClient(path="./chroma_db")

def create_collection(collection_name="vitechqa"):
    """
    Tạo collection mới — giống như tạo bảng trong database
    Nếu đã tồn tại thì xóa và tạo lại
    """
    # Xóa collection cũ nếu có
    try:
        client.delete_collection(collection_name)
        print(f"Đã xóa collection cũ: {collection_name}")
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # dùng cosine similarity
    )
    print(f"Tạo collection mới: {collection_name}")
    return collection


def add_chunks(collection, chunks):
    """
    Embed và lưu toàn bộ chunks vào ChromaDB
    """
    embeddings = embed_chunks(chunks)
    
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Đã lưu {len(chunks)} chunks vào ChromaDB.")


def retrieve(collection, query, top_k=5):
    """
    Tìm top_k chunks liên quan nhất với câu hỏi
    """
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results["documents"][0]  # list các chunks liên quan


# Test thử
if __name__ == "__main__":
    from .loader import load_pdf
    from .chunker import chunk_text
    
    # Load và chunk PDF
    text = load_pdf("data/CV.pdf")
    chunks = chunk_text(text)
    
    # Tạo collection và lưu vào ChromaDB
    collection = create_collection()
    add_chunks(collection, chunks)
    
    # Test retrieve
    print("\n=== TEST RETRIEVE ===")
    query = "GPA là bao nhiêu?"
    print(f"Câu hỏi: {query}")
    
    top_chunks = retrieve(collection, query, top_k=3)
    
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Chunk liên quan {i+1} ---")
        print(chunk[:300])  # in 300 ký tự đầu của mỗi chunk