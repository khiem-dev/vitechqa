import os
from dotenv import load_dotenv
from groq import Groq

# Tải API key từ file .env vào bộ nhớ hệ thống
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

model="llama-3.3-70b-versatile"
response = model.generate_content("Xin chào! RAG là gì? Giải thích trong 2 câu.")
print(response.text)
# .\rag-env\Scripts\activate Kích hoạt môi trường ảo