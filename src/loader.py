import pdfplumber

def load_pdf(pdf_path):
    """
    Đọc file PDF và trả về toàn bộ text
    """
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Tổng số trang: {len(pdf.pages)}")
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            if text:  # Một số trang có thể trống
                # Fix lỗi encoding tiếng Việt
                text = text.encode('utf-8', errors='ignore').decode('utf-8')
                full_text += text + "\n\n"
                
    print(f"Tổng số ký tự đọc được: {len(full_text)}")
    return full_text


# Test 
if __name__ == "__main__":
    text = load_pdf("data/Chuong_trinh_dao_tao.pdf")
    
    # In 500 ký tự đầu để kiểm tra
    print("\n--- 500 ký tự đầu tiên ---")
    print(text[:500])