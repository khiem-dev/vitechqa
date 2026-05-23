from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Prompt template — phần quan trọng nhất ảnh hưởng đến chất lượng
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
        response = model.generate_content(prompt)
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