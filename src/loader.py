# pdfplumber là thư viện đọc PDF — tốt hơn PyPDF2 vì
# xử lý được bảng biểu và tiếng Việt tốt hơn
import pdfplumber

def load_pdf(pdf_path):
    """
    Đọc file PDF và trả về toàn bộ text
    - PDF là file binary, không phải text (là các file có các ký tự liền mạch) — cần thư viện đặc biệt để đọc
    - Mỗi trang PDF (chữ được lưu như các điểm trong một không gian vector) được xử lý độc lập vì layout có thể khác nhau
    - Tiếng Việt hay bị lỗi encoding khi extract từ PDF
    """
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Tổng số trang: {len(pdf.pages)}")
        
        # extract_text() trả về chuỗi string (text) hoặc None nếu trang trống/ảnh
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            if text:  # Một số trang có thể trống
                # Fix lỗi encoding tiếng Việt
                text = text.encode('utf-8', errors='ignore').decode('utf-8') # Chuyển các string sang dạng Bytes, nếu gặp ký tự không thể đọc thì loại bỏ
                full_text += text + "\n\n" # Tạo ra một dòng trắng để ngăn cách giữa các Chunk
                
    print(f"Tổng số ký tự đọc được: {len(full_text)}")
    return full_text


# Test 
if __name__ == "__main__":
    text = load_pdf("data/Chuong_trinh_dao_tao.pdf")
    
    # In 500 ký tự đầu để kiểm tra
    print("\n--- 500 ký tự đầu tiên ---")
    print(text[:500])