import fitz  

def load_pdf(pdf_path):
    full_text = ""
    
    # Mở file PDF bằng PyMuPDF
    doc = fitz.open(pdf_path)
    print(f"Tổng số trang: {len(doc)}")
    
    for page in doc:
        text = page.get_text()
        if text:  
            full_text += text + "\n\n"
            
    print(f"Tổng số ký tự đọc được: {len(full_text)}")
    return full_text

if __name__ == "__main__":
    text = load_pdf("data/Chuong_trinh_dao_tao.pdf")
    print("\n--- 500 ký tự đầu tiên ---")
    print(text[:500])