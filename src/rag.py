import chromadb

try:
    from .vector_store import create_collection, add_chunks, retrieve
    from .loader import load_pdf
    from .chunker import chunk_text
    from .generator import generate_answer
except ImportError:
    from vector_store import create_collection, add_chunks, retrieve
    from loader import load_pdf
    from chunker import chunk_text
    from generator import generate_answer

# ============================================================
# CLASS RAGPipeline — Nhạc trưởng của toàn bộ hệ thống
# ============================================================
# Tại sao dùng Class thay vì function thông thường?
# → Class giữ state (self.collection, self.client) giữa các lần gọi
# → Không cần load lại ChromaDB mỗi lần user đặt câu hỏi
# → app.py tạo 1 instance duy nhất khi khởi động, tái sử dụng mãi
# → __init__: hàm khởi tạo, chạy 1 lần khi tạo object
# → self: tham chiếu đến chính object, dùng để lưu state
# → _method: convention đặt tên hàm private (chỉ dùng nội bộ trong class)
class RAGPipeline:
    def __init__(self, pdf_path=None, collection_name="vitechqa"):
        """
        Khởi tạo RAG pipeline.

        Tham số:
            pdf_path: đường dẫn file PDF
                      - Có giá trị → build index mới từ PDF (chạy lần đầu)
                      - None       → load index đã có sẵn (chạy từ lần 2)
            collection_name: tên collection trong ChromaDB
                             mặc định "vitechqa" — tên project
        """
        # Kết nối ChromaDB — đọc data đã lưu trên disk
        # path="./chroma_db" → thư mục lưu database
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./chroma_db")

        if pdf_path:
            # Có pdf_path → lần đầu chạy, cần build index từ đầu
            # Build index = load PDF + chunk + embed + lưu ChromaDB
            self.collection = self._build_index(pdf_path)
        else:
            # Load index đã có sẵn
            self.collection = self._load_index()

    def _build_index(self, pdf_path):
        """
        Build index từ đầu: PDF → chunks → embeddings → ChromaDB.

        Chạy 1 lần duy nhất khi thêm tài liệu mới.
        Sau khi build xong, data được lưu vào ./chroma_db/
        Những lần sau dùng _load_index() để load lại — không cần build lại.

        - "Index" trong RAG = cấu trúc dữ liệu cho phép tìm kiếm nhanh
        """
        print("Đang build index...")
        text = load_pdf(pdf_path)
        chunks = chunk_text(text)
        collection = create_collection(self.collection_name)
        add_chunks(collection, chunks)
        print(f"✅ Build xong: {len(chunks)} chunks")
        return collection

    def _load_index(self):
        try:
            collection = self.client.get_collection(self.collection_name)
            print(f"✅ Load index sẵn có: {collection.count()} chunks")
            return collection
        except Exception:
            raise Exception(
                "Chưa có index. Hãy chạy với pdf_path trước.\n"
                "Ví dụ: rag = RAGPipeline(pdf_path='data/file.pdf')"
            )

    def ask(self, question, top_k=5):
        # Bước 1: Tìm chunks liên quan
        chunks = retrieve(self.collection, question, top_k)

        # Bước 2: Sinh câu trả lời
        answer = generate_answer(chunks, question)

        return {
            "question": question,
            "answer": answer,
            "sources": chunks
        }


# Test thử
if __name__ == "__main__":
    # Lần đầu: build index từ PDF
    rag = RAGPipeline(pdf_path="data/Chuong_trinh_dao_tao.pdf")

    # Test hỏi đáp
    questions = [
        "Điều kiện tốt nghiệp là gì?",
        "Tổng số tín chỉ cần tích lũy là bao nhiêu?",
        "Khóa luận tốt nghiệp có bao nhiêu tín chỉ?"
    ]

    for q in questions:
        result = rag.ask(q)
        print(f"\n❓ {result['question']}")
        print(f"💬 {result['answer']}")
        print("-" * 50)