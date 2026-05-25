import gradio as gr
import time
from src.rag import RAGPipeline

# Khởi tạo RAG pipeline 1 lần khi app start
print("Đang khởi tạo RAG pipeline...")
rag = RAGPipeline()  # Load index đã có sẵn
print("✅ Sẵn sàng!")


def chat(question, history):
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

    history.append((question, result["answer"]))
    return "", history, sources_text


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

with gr.Blocks(title="ViTechQA", theme=gr.themes.Soft()) as demo:

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
                bubble_full_width=False
            )
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Nhập câu hỏi của bạn...",
                    label="",
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
    demo.launch(server_port=7861)