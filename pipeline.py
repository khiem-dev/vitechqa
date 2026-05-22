from src.loader import load_pdf
from src.chunker import chunk_text

def main():
    print("=== BƯỚC 1: Load PDF ===")
    text = load_pdf("data/CV.pdf")
    
    print("\n=== BƯỚC 2: Chunking ===")
    chunks = chunk_text(text)
    
    print(f"\n✅ Hoàn thành! Tạo được {len(chunks)} chunks từ tài liệu.")
    print("Sẵn sàng cho bước tiếp theo: Embedding.")

if __name__ == "__main__":
    main()