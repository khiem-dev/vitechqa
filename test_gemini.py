import os
from dotenv import load_dotenv
from google import genai

# Tải API key từ file .env vào bộ nhớ hệ thống
load_dotenv()

# CÚ PHÁP MỚI: Khởi tạo Client và truyền trực tiếp api_key của bạn vào đây
api_key = os.getenv("GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-2.5-flash", api_key=api_key)

response = model.generate_content("Xin chào! RAG là gì? Giải thích trong 2 câu.")
print(response.text)
# .\rag-env\Scripts\activate Kích hoạt môi trường ảo