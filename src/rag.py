import chromadb
from .loader import load_pdf
from .chunker import chunk_text
from .vector_store import create_collection, add_chunks, retrieve
from .generator import generate_answer


class RAGPipeline:
    def __init__(self, pdf_path=None, collection_name="vitechqa"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        if pdf_path:
            # Build index mới từ PDF
            self.collection = self._build_index(pdf_path)
        else:
            # Load index đã có
            self.collection = self._load_index()
    
    def _build_index(self, pdf_path):
        print("Building index...")
        text = load_pdf(pdf_path)
        chunks = chunk_text(text)
        collection = create_collection(self.collection_name)
        add_chunks(collection, chunks)
        print(f"✅ Index built: {len(chunks)} chunks")
        return collection
    
    def _load_index(self):
        try:
            return self.client.get_collection(self.collection_name)
        except:
            raise Exception("Chưa có index. Hãy chạy với pdf_path trước.")
    
    def ask(self, question, top_k=5):
        # Retrieve
        chunks = retrieve(self.collection, question, top_k)
        
        # Generate
        answer = generate_answer(chunks, question)
        
        return {
            "question": question,
            "answer": answer,
            "sources": chunks
        }


# Test thử
if __name__ == "__main__":
    rag = RAGPipeline(pdf_path="data/Chuong_trinh_dao_tao.pdf")
    
    result = rag.ask("Sinh viên phải tích lũy đủ bao nhiêu tín chỉ?")
    print(f"Câu hỏi: {result['question']}")
    print(f"Trả lời: {result['answer']}")
    print(f"\nNguồn: {result['sources'][0][:200]}")