from dotenv import load_dotenv
import os
from groq import Groq
# Groq là platform cung cấp LLM inference siêu nhanh (dùng chip LPU thay vì GPU)
# Ở đây dùng model LLaMA3 của Meta chạy trên Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Khởi tạo Groq client với API key
# Client này được tái sử dụng cho mọi request — không tạo lại mỗi lần gọi

PROMPT_TEMPLATE = """Bạn là trợ lý hỏi đáp tài liệu kỹ thuật tiếng Việt.
Nhiệm vụ: Trả lời câu hỏi DỰA TRÊN ngữ cảnh được cung cấp.

Quy tắc bắt buộc:
- Chỉ trả lời dựa trên thông tin có trong ngữ cảnh bên dưới.
- Nếu không tìm thấy thông tin, trả lời: "Tôi không tìm thấy thông tin này trong tài liệu."
- Không được bịa hoặc thêm thông tin ngoài ngữ cảnh.
- Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn.

NGỮ CẢNH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

"""
 1. SYSTEM INSTRUCTION ("Bạn là trợ lý..."):
    Định nghĩa vai trò và hành vi của LLM
    LLM sẽ đóng vai trò đó trong suốt cuộc hội thoại

 2. CONTEXT GROUNDING ("Chỉ trả lời dựa trên ngữ cảnh"):
    Đây là kỹ thuật quan trọng nhất trong RAG
    Không có dòng này → LLM dùng kiến thức của nó để trả lời
    → có thể sai hoàn toàn với tài liệu (hallucination)
    Có dòng này → LLM bị "neo" vào tài liệu được cung cấp

 3. FALLBACK INSTRUCTION ("Nếu không tìm thấy..."):
    Xử lý trường hợp câu hỏi nằm ngoài tài liệu
    Thay vì bịa câu trả lời, LLM thành thật nói không biết

 4. {context} và {question} là placeholder:
    Sẽ được thay thế bằng giá trị thật khi gọi .format()
    {context} = 5 chunks được retrieve từ ChromaDB
    {question} = câu hỏi của user
"""


def generate_answer(chunks, question):
    """
    Nhận các chunks liên quan và câu hỏi → gọi LLM → trả về câu trả lời.

    Đây là bước "G" (Generation) trong RAG pipeline:
    Retrieval (vector_store.py) → Generation (generator.py)

    Tham số:
        chunks: list các đoạn text được retrieve từ ChromaDB
                (top-5 chunks liên quan nhất với câu hỏi)
        question: câu hỏi của người dùng (string)

    Trả về:
        string: câu trả lời bằng tiếng Việt
    """
    # Bước 1: Ghép các chunks thành 1 đoạn context dài
    context = "\n\n---\n\n".join(chunks)
    # "\n\n---\n\n" là separator — phân cách rõ ràng giữa các nguồn khác nhau
    # Giúp LLM biết đây là các đoạn tài liệu riêng biệt, không phải 1 đoạn liền
    
    # Bước 2: Điền context và question vào template
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    # .format() thay thế {context} và {question} bằng giá trị thật
    # Kết quả là 1 chuỗi text hoàn chỉnh gửi cho LLM

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  
            messages=[
            # messages theo format ChatML (Chat Markup Language)
            # role "user" = tin nhắn từ người dùng
            # role "assistant" = tin nhắn từ AI (dùng khi có conversation history)
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  
            # temperature kiểm soát độ ngẫu nhiên của output:
            # temperature = 0.0 → deterministic, luôn ra cùng 1 kết quả
            # temperature = 0.1 → gần deterministic, ít ngẫu nhiên
            # temperature = 1.0 → sáng tạo, đa dạng nhưng kém ổn định
            # Với RAG dùng temperature thấp vì muốn câu trả lời nhất quán,
            # bám sát tài liệu — không cần sáng tạo
        )
        return response.choices[0].message.content
        # response.choices: list các câu trả lời (thường chỉ có 1)
        # [0]: lấy câu trả lời đầu tiên
        # .message.content: lấy phần text của câu trả lời
    except Exception as e:
        return f"Lỗi: {e}"


# Test thử
if __name__ == "__main__":
    test_chunks = [
        "Sinh viên phải tích lũy đủ 135 tín chỉ theo chương trình đào tạo.",
        "Khóa luận tốt nghiệp có 10 tín chỉ.",
        "Điều kiện tốt nghiệp: tích lũy đủ số tín chỉ."
    ]
    question = "Khóa luận tốt nghiệp có bao nhiêu tín chỉ?"
    answer = generate_answer(test_chunks, question)
    print(f"Câu hỏi: {question}")
    print(f"Trả lời: {answer}")