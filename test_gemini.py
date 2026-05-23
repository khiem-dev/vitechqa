import os
from dotenv import load_dotenv
# Nhập thư viện mới của Google
from google import genai

# Tải API key từ file .env vào bộ nhớ hệ thống
load_dotenv()

# CÚ PHÁP MỚI: Khởi tạo Client và truyền trực tiếp api_key của bạn vào đây
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# CÚ PHÁP MỚI: Gọi model thông qua client.models.generate_content
# (Đã cập nhật lên dòng model gemini-2.5-flash mới nhất, nhanh và thông minh hơn)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Xin chào! RAG là gì? Giải thích trong 2 câu."
)

print(response.text)

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser Cấp quyền để gọi môi trường ảo
# .\rag-env\Scripts\activate Kích hoạt môi trường ảo