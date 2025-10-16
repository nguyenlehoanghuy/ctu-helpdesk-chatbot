from modules.retriever import RetrievalSystem

def main():
    # Khởi tạo retrieval system
    retrieval_system = RetrievalSystem(embedding_model='bkai-foundation-models/vietnamese-bi-encoder', stopword_file='data/vietnamese-stopwords.txt')

    # 1. Load + tiền xử lý dữ liệu
    print("=== BƯỚC 1: LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU ===")
    retrieval_system.load_and_preprocess_data('data/ctu_chatbot_dataset.csv')

    # 2. Chia nhỏ documents thông minh
    print("\n=== BƯỚC 2: CHIA NHỎ DOCUMENTS ===")
    retrieval_system.chunk_documents_smart(max_chunk_size=512, overlap=225)

    # 3. Tạo embeddings
    print("\n=== BƯỚC 3: TẠO EMBEDDINGS ===")
    retrieval_system.create_embeddings()

    # 4. Xây dựng search indices (faiss, bm25, tfidf)
    print("\n=== BƯỚC 4: XÂY DỰNG SEARCH INDICES ===")
    retrieval_system.build_search_indices()

    # 5. Lưu hệ thống để sử dụng lại
    print("\n=== BƯỚC 5: LƯU HỆ THỐNG ===")
    retrieval_system.save_index('data/ctu_retrieval_system')

if __name__ == "__main__":
    system = main()
