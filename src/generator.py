from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


def generate_answer(chunks, question):
    context = "\n\n---\n\n".join(chunks)
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # model mạnh, miễn phí
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # thấp để câu trả lời ổn định
        )
        return response.choices[0].message.content
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