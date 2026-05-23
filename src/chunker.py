from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text):
    """
    Cắt text thành các đoạn nhỏ để embed
    chunk_size: 512 ký tự mỗi chunk
    chunk_overlap: 50 ký tự overlap giữa 2 chunk liên tiếp
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
        # Thứ tự ưu tiên: tách ở đoạn trống trước,
        # rồi xuống dòng, rồi dấu chấm, rồi khoảng trắng
    )
    
    chunks = splitter.split_text(text)
    print(f"Tổng số chunks: {len(chunks)}")
    return chunks


# Test thử
if __name__ == "__main__":
    from loader import load_pdf
    
    # Load PDF
    text = load_pdf("data/Chuong_trinh_dao_tao.pdf")
    
    # Chunk
    chunks = chunk_text(text)
    
    # In 3 chunks đầu để kiểm tra
    print("\n--- Chunk 0 ---")
    print(chunks[0])
    print("\n--- Chunk 1 ---")
    print(chunks[1])
    print("\n--- Chunk 2 ---")
    print(chunks[2])
    
    # Kiểm tra độ dài
    print(f"\nĐộ dài chunk ngắn nhất: {min(len(c) for c in chunks)}")
    print(f"Độ dài chunk dài nhất: {max(len(c) for c in chunks)}")
    print(f"Độ dài trung bình: {sum(len(c) for c in chunks) // len(chunks)}")