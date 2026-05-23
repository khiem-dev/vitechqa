import os
from dotenv import load_dotenv
# 1. Đổi sang thư viện thế hệ mới của Google
from google import genai

load_dotenv()

# 2. Khởi tạo Client theo chuẩn mới (Tự động nạp GEMINI_API_KEY từ file .env)
client = genai.Client()

# Prompt template — giữ nguyên cấu trúc rất tốt của bạn
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
    """
    Ghép chunks thành context, đưa vào Gemini để sinh câu trả lời
    """
    # Ghép các chunks lại thành context
    context = "\n\n---\n\n".join(chunks)
    
    # Điền vào template
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    
    try:
        # 3. Sử dụng cú pháp gọi hàm mới và dòng model gemini-2.5-flash tối ưu
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Lỗi khi gọi Gemini API: {e}"


# Test thử
if __name__ == "__main__":
    # Giả lập chunks được retrieve
    test_chunks = [
        "Sinh viên phải tích lũy đủ 135 tín chỉ theo chương trình đào tạo.",
        "Điều kiện tốt nghiệp: tích lũy đủ số tín chỉ của khối kiến thức giáo dục đại cương và giáo dục chuyên nghiệp như mô tả ở mục 6 và mục 7.",
        "Đồng thời thỏa các điều kiện tại Điều 17 Quy chế đào tạo trình độ đại học."
    ]
    
    question = "Điều kiện tốt nghiệp là gì?"
    answer = generate_answer(test_chunks, question)
    
    print(f"Câu hỏi: {question}")
    print(f"\nCâu trả lời:\n{answer}")