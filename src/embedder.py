import os
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import SentenceTransformer
import numpy as np

print("Đang load model bge-m3...", flush=True)
model = SentenceTransformer("BAAI/bge-m3")
print("Load model xong!", flush=True)

def embed_chunks(chunks):
    """
    Chuyển list các chunks thành list các vector embedding
    """
    print(f"Đang embed {len(chunks)} chunks...")
    
    embeddings = model.encode(
        chunks,
        batch_size=32,        # xử lý 32 chunks một lúc
        show_progress_bar=True # hiện thanh tiến độ
    )
    
    print(f"Xong! Mỗi chunk được chuyển thành vector {embeddings.shape[1]} chiều.")
    return embeddings


def embed_query(query):
    """
    Embed 1 câu hỏi để dùng khi retrieve
    """
    return model.encode([query])[0]


# Test 
if __name__ == "__main__":
    test_chunks = [
        "Môn ETC10013 có 2 tín chỉ, học kỳ 1.",
        "Điều kiện tốt nghiệp là tích lũy đủ 135 tín chỉ.",
        "Sinh viên phải học môn Triết học Mác-Lênin bắt buộc."
    ]
    
    embeddings = embed_chunks(test_chunks)
    print(f"\nShape của embeddings: {embeddings.shape}")
    print(f"Vector đầu tiên (5 số đầu): {embeddings[0][:5]}")