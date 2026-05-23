from src.rag import RAGPipeline

rag = RAGPipeline()

# Tự đặt 20 câu hỏi thực tế về tài liệu
# Chia làm 3 loại để test toàn diện
test_cases = [
    # Loại 1: Câu hỏi CÓ câu trả lời rõ ràng trong tài liệu
    {
        "question": "Tổng số tín chỉ cần tích lũy là bao nhiêu?",
        "expected_keyword": "135",  # từ khóa phải có trong câu trả lời
        "type": "factual"
    },
    {
        "question": "Điều kiện tốt nghiệp là gì?",
        "expected_keyword": "tín chỉ",
        "type": "factual"
    },
    {
        "question": "Khóa luận tốt nghiệp có bao nhiêu tín chỉ?",
        "expected_keyword": "10",
        "type": "factual"
    },
    {
        "question": "Môn học kỳ 1 gồm những môn nào?",
        "expected_keyword": "học kỳ",
        "type": "factual"
    },
    {
        "question": "Thời gian đào tạo là bao nhiêu năm?",
        "expected_keyword": "4",
        "type": "factual"
    },
    
    # Loại 2: Câu hỏi KHÔNG có trong tài liệu — phải từ chối
    {
        "question": "Học phí là bao nhiêu?",
        "expected_keyword": "không tìm thấy",
        "type": "out_of_scope"
    },
    {
        "question": "Giảng viên môn Toán là ai?",
        "expected_keyword": "không tìm thấy",
        "type": "out_of_scope"
    },
    
    # Loại 3: Câu hỏi liên quan gián tiếp
    {
        "question": "Sinh viên cần học bao nhiêu môn bắt buộc?",
        "expected_keyword": "bắt buộc",
        "type": "indirect"
    },
]

# Chạy test
print("=" * 60)
print("BÁO CÁO TEST CHẤT LƯỢNG RAG")
print("=" * 60)

passed = 0
failed = 0

for i, tc in enumerate(test_cases, 1):
    result = rag.ask(tc["question"])
    answer = result["answer"].lower()
    keyword = tc["expected_keyword"].lower()
    
    # Kiểm tra keyword có trong câu trả lời không
    is_pass = keyword in answer
    
    if is_pass:
        passed += 1
        status = "✅ PASS"
    else:
        failed += 1
        status = "❌ FAIL"
    
    print(f"\n[{i}] {status} | {tc['type'].upper()}")
    print(f"    Hỏi: {tc['question']}")
    print(f"    Cần có: '{tc['expected_keyword']}'")
    print(f"    Trả lời: {result['answer'][:150]}...")

print("\n" + "=" * 60)
print(f"KẾT QUẢ: {passed}/{len(test_cases)} câu đúng")
print(f"Tỷ lệ: {passed/len(test_cases)*100:.0f}%")
print("=" * 60)