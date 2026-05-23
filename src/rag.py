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


class RAGPipeline:
    def __init__(self, pdf_path=None, collection_name="vitechqa"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./chroma_db")

        if pdf_path:
            # Build index mới từ PDF
            self.collection = self._build_index(pdf_path)
        else:
            # Load index đã có sẵn
            self.collection = self._load_index()

    def _build_index(self, pdf_path):
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