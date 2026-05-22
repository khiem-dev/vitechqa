import pdfplumber

def load_pdf(pdf_path):
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Tổng số trang: {len(pdf.pages)}")
        
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            if text:  
                text = text.encode('utf-8', errors='ignore').decode('utf-8')
                full_text += text + "\n\n"
                
    print(f"Tổng số ký tự đọc được: {len(full_text)}")
    return full_text


if __name__ == "__main__":
    text = load_pdf("data/CV.pdf")
    
    print("\n--- 500 ký tự đầu tiên ---")
    print(text[:500])