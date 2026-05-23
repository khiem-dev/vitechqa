import gradio as gr
from src.rag import RAGPipeline
import time

# Khởi tạo RAG pipeline 1 lần khi app start
print("Đang khởi tạo RAG pipeline...")
rag = RAGPipeline()  # Load index đã có từ ngày 3-4
print("✅ Sẵn sàng!")


def chat(question, history):
    """
    Hàm xử lý câu hỏi — Cập nhật cấu trúc dict theo chuẩn Gradio mới
    """
    if not question.strip():
        # Định dạng history lúc này là list các dict, xóa hội thoại trả về list rỗng []
        return "", history, "Vui lòng nhập câu hỏi."
    
    # Gọi RAG pipeline
    start_time = time.time()
    result = rag.ask(question, top_k=5)
    elapsed = time.time() - start_time
    
    answer = result["answer"]
    sources = result["sources"]
    
    # Format sources để hiển thị
    sources_text = f"⏱ Thời gian xử lý: {elapsed:.2f}s\n\n"
    sources_text += "📄 Nguồn tham khảo:\n"
    sources_text += "=" * 40 + "\n"
    for i, src in enumerate(sources[:3], 1):
        sources_text += f"\n[Nguồn {i}]\n{src[:300]}...\n"
        sources_text += "-" * 30 + "\n"
    
    # SỬA TẠI ĐÂY: Thêm tin nhắn của User và Assistant theo chuẩn định dạng mới
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    return "", history, sources_text


def clear_chat():
    return [], ""

# Một số câu hỏi mẫu để demo
example_questions = [
    "Điều kiện tốt nghiệp là gì?",
    "Tổng số tín chỉ cần tích lũy là bao nhiêu?",
    "Môn học bắt buộc trong học kỳ 1 là những môn nào?",
    "Khóa luận tốt nghiệp có bao nhiêu tín chỉ?",
    "Các môn tự chọn trong chương trình đào tạo gồm những môn nào?"
]


# Build giao diện (Đã lược bỏ theme, css tại đây để tránh lỗi UserWarning)
with gr.Blocks(title="ViTechQA") as demo:
    
    # Header
    gr.Markdown(
        """
        # 🤖 ViTechQA
        ### Hệ thống Hỏi đáp Tài liệu Kỹ thuật Tiếng Việt
        Hỏi bất kỳ câu hỏi nào về tài liệu — hệ thống sẽ tìm và trả lời dựa trên nội dung tài liệu gốc.
        """,
        elem_classes="title"
    )
    
    with gr.Row():
        # Cột trái — Chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Hội thoại",
                height=450
                # Đã loại bỏ 'bubble_full_width=False' gây lỗi crash hệ thống ở đây
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Nhập câu hỏi của bạn...",
                    label="",
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button(
                    "Gửi 🚀",
                    variant="primary",
                    scale=1
                )
            
            clear_btn = gr.Button("Xóa hội thoại 🗑️", variant="secondary")
            
            # Câu hỏi mẫu
            gr.Markdown("**💡 Câu hỏi mẫu — click để dùng:**")
            gr.Examples(
                examples=example_questions,
                inputs=question_input,
                label=""
            )
        
        # Cột phải — Sources
        with gr.Column(scale=2):
            sources_output = gr.Textbox(
                label="📄 Nguồn tham khảo",
                lines=20,
                interactive=False,
                placeholder="Nguồn tài liệu sẽ hiển thị ở đây sau khi bạn đặt câu hỏi..."
            )
    
    # Footer info
    gr.Markdown(
        """
        ---
        **Tech stack:** Python · LangChain · ChromaDB · BAAI/bge-m3 · Gemini API · Gradio  
        **Tác giả:** Nguyễn Lê Gia Khiêm · HCMUS · 2025
        """
    )
    
    # Kết nối sự kiện
    submit_btn.click(
        fn=chat,
        inputs=[question_input, chatbot],
        outputs=[question_input, chatbot, sources_output]
    )
    
    # Enter để submit
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
        share=False,    # Tạo link public tạm thời cho đồ án
        server_port=7861,
        theme=gr.themes.Soft(), # Đã chuyển tham số theme xuống đúng hàm launch() theo chuẩn mới
        css="""
            .title { text-align: center; margin-bottom: 10px; }
            .subtitle { text-align: center; color: #666; margin-bottom: 20px; }
            footer { display: none !important; }
        """ # Đã chuyển tham số css xuống đúng hàm launch()
    )