import gradio as gr
# Gradio là thư viện tạo web UI cho AI/ML project
# Chỉ cần ~20 dòng code để có giao diện web hoàn chỉnh
# Không cần biết HTML/CSS/JavaScript
# Tự động tạo server local và có thể deploy lên HuggingFace Spaces
import time
# Dùng để đo thời gian xử lý mỗi câu hỏi
# Giúp user biết hệ thống đang hoạt động và nhanh hay chậm
from src.rag import RAGPipeline
# Import class RAGPipeline từ src/rag.py
# "src.rag" = file rag.py trong thư mục src
# RAGPipeline chứa toàn bộ logic: retrieve + generate

# Khởi tạo RAG pipeline 1 lần khi app start
print("Đang khởi tạo RAG pipeline...")
rag = RAGPipeline()  # Load index đã có sẵn
print("✅ Sẵn sàng!")

# ============================================================
# HÀM XỬ LÝ CHÍNH — Gradio gọi hàm này mỗi khi user submit
# ============================================================
def chat(question, history):
    """
    Xử lý câu hỏi của user và trả về câu trả lời.

    Gradio tự động truyền các tham số vào hàm này:
        question: text user vừa nhập vào Textbox
        history: list các cặp (câu hỏi, câu trả lời) trước đó
                 format: [("hỏi 1", "trả lời 1"), ("hỏi 2", "trả lời 2")]

    Trả về tuple 3 giá trị — Gradio map vào outputs theo thứ tự:
        "": xóa nội dung Textbox sau khi submit
        history: cập nhật Chatbot với câu hỏi/trả lời mới
        sources_text: cập nhật Textbox nguồn tham khảo
    """
    if not question.strip():
        return "", history, "Vui lòng nhập câu hỏi."

    start_time = time.time()
    result = rag.ask(question, top_k=5)
    elapsed = time.time() - start_time

    # Format sources
    sources_text = f"⏱ Thời gian: {elapsed:.2f}s\n\n"
    sources_text += "📄 Nguồn tham khảo:\n"
    sources_text += "=" * 40 + "\n"
    for i, src in enumerate(result["sources"][:3], 1):
        sources_text += f"\n[Nguồn {i}]\n{src[:300]}\n"
        sources_text += "-" * 30 + "\n"

    new_history = history + [
    {"role": "user", "content": question},
    {"role": "assistant", "content": result["answer"]}
    ]
    return "", new_history, sources_text


def clear_chat():
    return [], ""


# Câu hỏi mẫu
examples = [
    "Điều kiện tốt nghiệp là gì?",
    "Tổng số tín chỉ cần tích lũy là bao nhiêu?",
    "Môn học kỳ 1 gồm những môn nào?",
    "Khóa luận tốt nghiệp có bao nhiêu tín chỉ?",
    "Các môn tự chọn gồm những môn nào?"
]

with gr.Blocks(title="ViTechQA") as demo:

    gr.Markdown("""
    # 🤖 ViTechQA
    ### Hệ thống Hỏi đáp Tài liệu Kỹ thuật Tiếng Việt
    Đặt câu hỏi về tài liệu — hệ thống trả lời dựa trên nội dung gốc, không hallucinate.
    """)

    with gr.Row():

        # Cột trái — Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Hội thoại",
                height=450,
                value=[],
            )
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Nhập câu hỏi của bạn...",
                    label=False,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Gửi 🚀", variant="primary", scale=1)

            clear_btn = gr.Button("Xóa hội thoại 🗑️", variant="secondary")

            gr.Markdown("**💡 Câu hỏi mẫu:**")
            gr.Examples(examples=examples, inputs=question_input, label="")

        # Cột phải — Sources
        with gr.Column(scale=2):
            sources_output = gr.Textbox(
                label="📄 Nguồn tham khảo",
                lines=22,
                interactive=False,
                placeholder="Nguồn sẽ hiển thị ở đây..."
            )

    gr.Markdown("""
    ---
    **Tech stack:** Python · LangChain · ChromaDB · BAAI/bge-m3 · Groq LLaMA3 · Gradio
    **Tác giả:** Nguyễn Lê Gia Khiêm · HCMUS · 2026
    """)

    # Sự kiện
    submit_btn.click(
        fn=chat,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot, sources_output]
    )
    question_input.submit(
        fn=chat,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot, sources_output]
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, sources_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7861,
        theme=gr.themes.Soft()
    )