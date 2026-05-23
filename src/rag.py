import sys
from pathlib import Path

# Cho phép chạy `python src/rag.py` và import `from src.rag` trong app.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chromadb
from src.loader import load_pdf
from src.chunker import chunk_text
from src.vector_store import create_collection, add_chunks, retrieve
from src.generator import generate_answer

DEFAULT_PDF = _ROOT / "data" / "Chuong_trinh_dao_tao.pdf"
CHROMA_PATH = _ROOT / "chroma_db"


class RAGPipeline:
    def __init__(self, pdf_path=None, collection_name="vitechqa"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        if pdf_path:
            path = Path(pdf_path)
            if not path.is_absolute():
                path = _ROOT / path
            self.collection = self._build_index(path)
        else:
            self.collection = self._load_index()

    def _build_index(self, pdf_path):
        print("Building index...")
        text = load_pdf(str(pdf_path))
        chunks = chunk_text(text)
        collection = create_collection(self.collection_name)
        add_chunks(collection, chunks)
        print(f"Index built: {len(chunks)} chunks")
        return collection

    def _load_index(self):
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            raise RuntimeError(
                "Chua co index. Chay: python src/rag.py --build"
            ) from None

    def ask(self, question, top_k=5):
        chunks = retrieve(self.collection, question, top_k)
        answer = generate_answer(chunks, question)

        return {
            "question": question,
            "answer": answer,
            "sources": chunks,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RAG pipeline")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build index tu PDF (chi chay lan dau hoac khi can rebuild)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Duong dan PDF khi dung --build",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Điều kiện tốt nghiệp là gì?",
        help="Câu hỏi test",
    )
    args = parser.parse_args()

    if args.build:
        rag = RAGPipeline(pdf_path=args.pdf)
    else:
        rag = RAGPipeline()

    result = rag.ask(args.question)
    print(f"Cau hoi: {result['question']}")
    print(f"Tra loi: {result['answer']}")
    if result["sources"]:
        print(f"\nNguon: {result['sources'][0][:200]}")
