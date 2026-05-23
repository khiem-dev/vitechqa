from sentence_transformers import SentenceTransformer
import torch

_model = None


def _get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang load model bge-m3 trên thiết bị: {device.upper()}...")
        _model = SentenceTransformer("BAAI/bge-m3", device=device)
        print("Load model xong!")
    return _model


def embed_chunks(chunks):
    """
    Chuyển list các chunks thành list các vector embedding
    """
    print(f"Đang embed {len(chunks)} chunks...")

    embeddings = _get_model().encode(
        chunks,
        batch_size=32,
        show_progress_bar=True
    )
    
    print(f"Xong! Mỗi chunk được chuyển thành vector {embeddings.shape[1]} chiều.")
    return embeddings


def embed_query(query):
    """
    Embed 1 câu hỏi để dùng khi retrieve
    """
    return _get_model().encode([query])[0]