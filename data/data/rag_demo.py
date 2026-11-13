"""
====================================================================
DEMO RAG ENGINE CHO DỮ LIỆU TỔ HỢP KHÁCH SẠN / RESORT
Sử dụng trực tiếp tệp: hotel_rag_knowledge_base.json
Chạy độc lập, không yêu cầu cài đặt thêm thư viện ngoài (Pure Python)
====================================================================
"""

import json
import math
import os
import re

DATA_PATH = os.path.join(os.path.dirname(__file__), "hotel_rag_knowledge_base.json")

def tokenize(text):
    text = text.lower()
    # Tách từ tiếng Việt đơn giản theo khoảng trắng và bỏ ký tự đặc biệt
    words = re.findall(r'\b\w+\b', text)
    return words

class SimpleRAGRetriever:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        self.doc_tokens = [tokenize(doc["content"] + " " + doc["title"] + " " + doc["category"]) for doc in self.documents]
        self.N = len(self.documents)
        
        # Tính DF (Document Frequency)
        self.df = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.df[t] = self.df.get(t, 0) + 1
                
        # Tính IDF (Inverse Document Frequency)
        self.idf = {t: math.log((self.N + 1) / (df_val + 1)) + 1 for t, df_val in self.df.items()}

    def search(self, query, top_k=3):
        query_tokens = tokenize(query)
        scores = []

        for idx, tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = len(tokens)
            token_counts = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            for qt in query_tokens:
                if qt in token_counts:
                    tf = token_counts[qt] / doc_len
                    idf = self.idf.get(qt, 1.0)
                    score += tf * idf

            scores.append((score, self.documents[idx]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

def answer_query(retriever, query):
    print("\n=======================================================")
    print(f"❓ CÂU HỎI: \"{query}\"")
    print("=======================================================")
    results = retriever.search(query, top_k=2)

    print("🔎 CÁC ĐOẠN TRI THỨC TRÍCH XUẤT ĐƯỢC (RETRIEVED CONTEXT):")
    for rank, (score, doc) in enumerate(results, 1):
        print(f"\n[{rank}] Tiêu đề: {doc['title']} (Score: {score:.4f})")
        print(f"    Phân loại: {doc['category']}")
        print(f"    Nội dung: {doc['content']}")

    print("\n💡 TỔNG HỢP CÂU TRẢ LỜI CHO NGƯỜI DÙNG:")
    if results and results[0][0] > 0.05:
        top_doc = results[0][1]
        print(f"Dựa trên cơ sở dữ liệu nghỉ dưỡng: {top_doc['title']}.\n"
              f"{top_doc['content']}")
    else:
        print("Xin lỗi, hiện tại hệ thống chưa tìm thấy thông tin phù hợp với yêu cầu của bạn.")

if __name__ == "__main__":
    retriever = SimpleRAGRetriever(DATA_PATH)
    print(f"-> Đã nạp thành công {retriever.N} chunks kiến thức vào RAG Retriever.")

    # Thử nghiệm với các câu hỏi thực tế của khách hàng
    test_queries = [
        "Phòng President giá bao nhiêu tiền một đêm và có tiện ích gì?",
        "Khách sạn có tour lặn biển ngắm san hô không?",
        "Quy định đặt cọc tiền phòng và chính sách hủy phòng thế nào?"
    ]

    for q in test_queries:
        answer_query(retriever, q)

