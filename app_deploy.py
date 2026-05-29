import gradio as gr
import os
import time
from src.rag import RAGPipeline

# Trên HuggingFace, API key lưu trong Secrets (không dùng .env)
api_key = os.environ.get("GROQ_API_KEY")

rag = None 
INDEX_BUILT = False

def build_index_from_upload(pdf_file):
    """
    Build index từ file PDF người dùng upload
    """
    global rag, INDEX_BUILT
    
    if pdf_file is None:
        return "⚠️ Vui lòng upload file PDF trước."
    
    try:
        from src.loader import load_pdf
        from src.chunker import chunk_text
        from src.vector_store import create_collection, add_chunks
        import chromadb
        
        client = chromadb.PersistentClient(path="./chroma_db")
        
        text = load_pdf(pdf_file.name)
        chunks = chunk_text(text)
        collection = create_collection()
        add_chunks(collection, chunks)
        
        rag = RAGPipeline()
        INDEX_BUILT = True
        
        return f"✅ Đã index xong {len(chunks)} chunks từ tài liệu!"
    except Exception as e:
        return f"❌ Lỗi: {e}"


def chat(question, history):
    global rag, INDEX_BUILT
    
    if not question.strip():
        return "", history, "Vui lòng nhập câu hỏi."
    
    if not INDEX_BUILT or rag is None:
        history.append((question, "⚠️ Vui lòng upload và index tài liệu PDF trước!"))
        return "", history, ""
    
    start_time = time.time()
    result = rag.ask(question, top_k=5)
    elapsed = time.time() - start_time
    
    sources_text = f"⏱ {elapsed:.2f}s\n\n📄 Nguồn tham khảo:\n{'='*40}\n"
    for i, src in enumerate(result["sources"][:3], 1):
        sources_text += f"\n[{i}] {src[:300]}...\n{'-'*30}\n"
    
    new_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]}
    ]
    return "", new_history, sources_text


def clear_all():
    return [], ""


with gr.Blocks(title="ViTechQA") as demo:
    gr.Markdown("""
    # 🤖 ViTechQA — Vietnamese Technical Document Q&A
    Upload tài liệu PDF tiếng Việt và hỏi bất kỳ câu hỏi nào!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_upload = gr.File(
                label="📁 Upload tài liệu PDF",
                file_types=[".pdf"]
            )
            index_btn = gr.Button("🔧 Build Index", variant="primary")
            index_status = gr.Textbox(
                label="Trạng thái",
                interactive=False,
                lines=2
            )
            gr.Markdown("""
            **Hướng dẫn:**
            1. Upload file PDF
            2. Click "Build Index"
            3. Chờ index xong
            4. Bắt đầu đặt câu hỏi
            """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400)
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="Nhập câu hỏi...",
                    label="", scale=4, container=False
                )
                submit_btn = gr.Button("Gửi 🚀", variant="primary", scale=1)
            clear_btn = gr.Button("Xóa 🗑️", variant="secondary")
        
        with gr.Column(scale=2):
            sources_output = gr.Textbox(
                label="📄 Nguồn tham khảo",
                lines=18, interactive=False
            )
    
    # Events
    index_btn.click(build_index_from_upload, inputs=[pdf_upload], outputs=[index_status])
    submit_btn.click(chat, inputs=[question_input, chatbot], outputs=[question_input, chatbot, sources_output])
    question_input.submit(chat, inputs=[question_input, chatbot], outputs=[question_input, chatbot, sources_output])
    clear_btn.click(clear_all, outputs=[chatbot, sources_output])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())