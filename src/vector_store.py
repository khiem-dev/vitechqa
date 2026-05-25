import chromadb
# ChromaDB là vector database — lưu trữ và tìm kiếm vector embedding
# "vector database" khác "database thông thường":
# - Database thường: tìm kiếm bằng từ khóa chính xác (WHERE name = 'X')
# - Vector database: tìm kiếm bằng độ tương đồng ngữ nghĩa
#   (tìm vector nào GẦN NHẤT với vector câu hỏi)


def _import_embedder():
    try:
        from .embedder import embed_chunks, embed_query
    except ImportError:
        from embedder import embed_chunks, embed_query
    return embed_chunks, embed_query

# Khởi tạo ChromaDB lưu trên disk
# PersistentClient = lưu data xuống disk, không mất khi tắt máy
# (khác với InMemoryClient chỉ lưu trong RAM)
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
    
    """
    - Tạo collection trong ChromaDB — giống như tạo bảng trong SQL database.
    - Collection chứa các documents (chunks) và embedding tương ứng
    - metadata={"hnsw:space": "cosine"}: dùng cosine similarity để đo khoảng cách
    - Cosine similarity đo góc giữa 2 vector (0 = vuông góc, 1 = cùng hướng)
    - Khác với Euclidean distance đo khoảng cách tuyệt đối
    - Cosine tốt hơn cho text vì không bị ảnh hưởng bởi độ dài văn bản
    """
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # dùng cosine similarity thay vì dùng L2
    )
    print(f"Tạo collection mới: {collection_name}")
    return collection


def add_chunks(collection, chunks):
    """
    Embed và lưu toàn bộ chunks vào ChromaDB
    - Mỗi document trong ChromaDB có 3 thành phần:
      + documents: text gốc (để trả về cho user đọc)
      + embeddings: vector số (để tính similarity)
      + ids: định danh duy nhất (để update/delete sau này)
    - .tolist() chuyển numpy array → Python list vì ChromaDB không nhận numpy
    """
    embed_chunks, _ = _import_embedder()
    embeddings = embed_chunks(chunks)
    
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"Đã lưu {len(chunks)} chunks vào ChromaDB.")


def retrieve(collection, query, top_k):
    """
    Tìm top_k chunks liên quan nhất với câu hỏi.
    - Đây là bước "R" (Retrieval) trong RAG
    - Quy trình:
      1. Embed câu hỏi → query_vector (1024 số)
      2. Tính cosine similarity giữa query_vector và TẤT CẢ chunk vectors
      3. Trả về top_k chunks có similarity cao nhất
    - top_k=5: lấy 5 chunks liên quan nhất
      → đủ context cho LLM nhưng không quá dài
    - Chất lượng retrieval ảnh hưởng trực tiếp đến chất lượng câu trả lời
      "garbage in, garbage out" — retrieve sai thì LLM trả lời sai
    """
    _, embed_query = _import_embedder()
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results["documents"][0]  # list các chunks liên quan


# Test thử
if __name__ == "__main__":
    try:
        from .loader import load_pdf
        from .chunker import chunk_text
    except ImportError:
        from loader import load_pdf
        from chunker import chunk_text
    
    # Load và chunk PDF
    text = load_pdf("data/Chuong_trinh_dao_tao.pdf")
    chunks = chunk_text(text)
    
    # Tạo collection và lưu vào ChromaDB
    collection = create_collection()
    add_chunks(collection, chunks)
    
    # Test retrieve
    print("\n=== TEST RETRIEVE ===")
    query = "Khóa luận tốt nghiệp bao nhiêu tín chỉ?"
    print(f"Câu hỏi: {query}")
    
    top_chunks = retrieve(collection, query, top_k=3)
    
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Chunk liên quan {i+1} ---")
        print(chunk[:300])  # in 300 ký tự đầu của mỗi chunk