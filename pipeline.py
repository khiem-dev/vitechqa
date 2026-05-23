import chromadb
from src.loader import load_pdf
from src.chunker import chunk_text
from src.vector_store import create_collection, add_chunks, retrieve
from src.generator import generate_answer

# Kết nối lại ChromaDB đã build từ ngày 3
client = chromadb.PersistentClient(path="./chroma_db")


def build_index():
    """
    Chạy 1 lần để build index — nếu đã build rồi thì bỏ qua
    """
    text = load_pdf("data/Chuong_trinh_dao_tao.pdf")
    chunks = chunk_text(text)
    collection = create_collection()
    add_chunks(collection, chunks)
    print("✅ Build index xong!")
    return collection


def get_collection():
    """
    Lấy collection đã có — không cần build lại
    """
    try:
        return client.get_collection("vitechqa")
    except:
        print("Chưa có index, đang build...")
        return build_index()


def ask(question, top_k=7):
    """
    Hàm chính: nhận câu hỏi → retrieve → generate → trả lời
    """
    print(f"\n🔍 Câu hỏi: {question}")
    
    # Bước 1: Lấy collection
    collection = get_collection()
    
    # Bước 2: Retrieve chunks liên quan
    chunks = retrieve(collection, question, top_k=top_k)
    print(f"📄 Tìm được {len(chunks)} chunks liên quan")
    
    # Bước 3: Generate câu trả lời
    answer = generate_answer(chunks, question)
    
    return answer, chunks


# Test end-to-end
if __name__ == "__main__":
    questions = [
        "Điều kiện tốt nghiệp là gì?",
        "Tổng số tín chỉ cần tích lũy là bao nhiêu?",
        "Môn học bắt buộc trong học kỳ 1 là những môn nào?",
        "Khóa luận tốt nghiệp có bao nhiêu tín chỉ?"
    ]
    
    for q in questions:
        answer, chunks = ask(q)
        print(f"\n💬 Trả lời:\n{answer}")
        print("\n" + "="*60)