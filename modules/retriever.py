import os
import pickle
import logging
import re
from typing import List, Dict

import numpy as np
import torch
import pandas as pd
import faiss
from underthesea import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

class RetrievalSystem:
    def __init__(self, embedding_model='bkai-foundation-models/vietnamese-bi-encoder', stopword_file='vietnamese-stopwords.txt'):
        """
        Khởi tạo hệ thống retrieval cải tiến cho CTU chatbot

        Args:
            embedding_model: Tên model để tạo embeddings (dùng Vietnamese model)
            stopword_file: Đường dẫn tới file chứa danh sách stopwords
        """
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load Vietnamese-optimized model
        self.embedding_model = SentenceTransformer(embedding_model)
        if self.device == 'cuda':
            self.embedding_model = self.embedding_model.cuda()

        self.semantic_index = None
        self.bm25_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self.documents = []
        self.processed_documents = []
        self.document_embeddings = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Vietnamese stopwords (basic set)
        # Load Vietnamese stopwords từ file
        try:
            with open(stopword_file, 'r', encoding='utf-8') as f:
                self.vietnamese_stopwords = set(
                    line.strip().lower() for line in f if line.strip()
                )
            self.logger.info(f"Loaded {len(self.vietnamese_stopwords)} Vietnamese stopwords từ '{stopword_file}'")
        except FileNotFoundError:
            self.logger.warning(f"Không tìm thấy file stopwords tại '{stopword_file}'. Dùng danh sách mặc định.")
            self.vietnamese_stopwords = {
                'và', 'của', 'có', 'là', 'trong', 'với', 'để', 'một', 'các', 'được',
                'này', 'đó', 'những', 'như', 'từ', 'về', 'theo', 'sau', 'trước',
                'khi', 'bằng', 'tại', 'cho', 'đã', 'sẽ', 'không', 'chỉ', 'cũng',
                'thì', 'nếu', 'mà', 'nhưng', 'vì', 'do', 'nên', 'hay', 'hoặc'
            }

    def load_and_preprocess_data(self, file_path: str) -> List[Dict]:
        """
        Đọc và tiền xử lý dữ liệu từ CSV file với nhiều cải tiến
        """
        self.logger.info("Đang đọc dữ liệu từ CSV file...")

        # Đọc dữ liệu
        df = pd.read_csv(file_path)

        # Kiểm tra cấu trúc dữ liệu
        if 'content' in df.columns:
            content_col = 'content'
        elif 'context' in df.columns:
            content_col = 'context'
        else:
            raise ValueError("File phải chứa cột 'content' hoặc 'context'")

        self.logger.info(f"Tổng số record: {len(df)}")

        # Lấy unique content và loại bỏ duplicate
        unique_contents = df[content_col].dropna().drop_duplicates().tolist()
        self.logger.info(f"Số unique content sau khi loại bỏ duplicate: {len(unique_contents)}")

        # Tiền xử lý content với nhiều kỹ thuật
        processed_docs = []
        for idx, content in enumerate(unique_contents):
            # Làm sạch text
            clean_content = content # self._clean_text_advanced(content)

            if len(clean_content.strip()) < 10:  # Skip quá ngắn
                continue

            # Tạo multiple representations
            doc_data = {
                'id': f'doc_{idx}',
                'content': clean_content,
                'original_content': content,
                'length': len(clean_content),
                'word_count': len(clean_content.split()),
                'sentences': sent_tokenize(clean_content),
                'keywords': self._extract_keywords(clean_content),
                'processed_for_search': self._process_for_search(clean_content)
            }

            processed_docs.append(doc_data)

        self.documents = processed_docs
        self.logger.info(f"Hoàn thành tiền xử lý {len(processed_docs)} documents")

        return processed_docs

    def _clean_text_advanced(self, text: str) -> str:
        """
        Làm sạch text content với nhiều kỹ thuật
        """
        if not isinstance(text, str):
            return ""

        # Chuẩn hóa encoding
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # Loại bỏ ký tự xuống dòng và thay thế bằng space
        clean_text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

        # Loại bỏ ký tự đặc biệt nhưng giữ dấu câu quan trọng
        clean_text = re.sub(r'[^\w\s\.\,\:\;\?\!\-\(\)]', ' ', clean_text)

        # Loại bỏ nhiều space liền nhau
        clean_text = re.sub(r'\s+', ' ', clean_text)

        # Trim whitespace
        clean_text = clean_text.strip()

        return clean_text

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Trích xuất keywords quan trọng từ text
        """
        # Word tokenize
        words = word_tokenize(text.lower())

        # Loại bỏ stopwords và từ quá ngắn
        keywords = [word for word in words
                   if word not in self.vietnamese_stopwords
                   and len(word) > 2
                   and word.isalnum()]

        # Đếm tần suất và lấy top keywords
        from collections import Counter
        counter = Counter(keywords)
        top_keywords = [word for word, count in counter.most_common(10)]

        return top_keywords

    def _process_for_search(self, text: str) -> str:
        """
        Xử lý text để tối ưu cho search (word segmentation)
        """
        try:
            # Sử dụng underthesea để tách từ tiếng Việt
            words = word_tokenize(text)
            return ' '.join(words)
        except:
            # Fallback nếu underthesea lỗi
            return text

    def chunk_documents_smart(self, max_chunk_size: int = 300, overlap: int = 100) -> List[Dict]:
        """
        Chia nhỏ documents thông minh theo câu và đoạn văn
        """
        if not self.documents:
            raise ValueError("Chưa load dữ liệu. Hãy gọi load_and_preprocess_data() trước.")

        chunked_docs = []

        for doc in self.documents:
            content = doc['content']
            sentences = doc['sentences']

            # Nếu document ngắn hơn max_chunk_size, giữ nguyên
            if len(content) <= max_chunk_size:
                chunk_data = doc.copy()
                chunk_data.update({
                    'id': doc['id'] + '_chunk_0',
                    'parent_id': doc['id'],
                    'chunk_index': 0,
                    'content': content
                })
                chunked_docs.append(chunk_data)
            else:
                # Chia thành chunks theo câu
                chunks = self._split_by_sentences(sentences, max_chunk_size, overlap)
                for i, chunk_content in enumerate(chunks):
                    chunk_data = doc.copy()
                    chunk_data.update({
                        'id': doc['id'] + f'_chunk_{i}',
                        'parent_id': doc['id'],
                        'chunk_index': i,
                        'content': chunk_content,
                        'length': len(chunk_content),
                        'word_count': len(chunk_content.split()),
                        'processed_for_search': self._process_for_search(chunk_content)
                    })
                    chunked_docs.append(chunk_data)

        self.processed_documents = chunked_docs
        self.logger.info(f"Tổng số chunk sau khi chia: {len(chunked_docs)}")
        return chunked_docs

    def _split_by_sentences(self, sentences: List[str], max_size: int, overlap: int) -> List[str]:
        """
        Chia text thành chunks theo câu với overlap thông minh
        """
        chunks = []
        current_chunk = ""
        overlap_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Nếu thêm câu này vào chunk hiện tại mà vẫn < max_size
            if len(current_chunk + " " + sentence) <= max_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Lưu chunk hiện tại
                if current_chunk:
                    chunks.append(current_chunk)

                    # Chuẩn bị overlap cho chunk tiếp theo
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence

        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Lấy phần overlap từ cuối text
        """
        if len(text) <= overlap_size:
            return text
        else:
            # Tìm vị trí space gần nhất để không cắt giữa từ
            start_pos = len(text) - overlap_size
            while start_pos > 0 and text[start_pos] != ' ':
                start_pos -= 1
            return text[start_pos:].strip()

    def create_embeddings(self, documents: List[Dict] = None) -> np.ndarray:
        """
        Tạo embeddings với Vietnamese tokenization
        """
        if documents is None:
            documents = self.processed_documents if self.processed_documents else self.documents

        if not documents:
            raise ValueError("Không có document nào để tạo embedding")

        self.logger.info(f"Đang tạo embeddings cho {len(documents)} documents...")

        # Sử dụng processed text cho embedding
        texts = []
        for doc in documents:
            if 'processed_for_search' in doc:
                texts.append(doc['processed_for_search'])
            else:
                texts.append(doc['content'])

        # Tạo embeddings với batch processing
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Chuẩn hóa để dùng cosine similarity
            )
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)
        self.document_embeddings = embeddings
        self.logger.info(f"Hoàn thành tạo embeddings. Shape: {embeddings.shape}")

        return embeddings

    def build_search_indices(self, embeddings: np.ndarray = None):
        """
        Xây dựng cả semantic và keyword search indices
        """
        if embeddings is None:
            embeddings = self.document_embeddings

        if embeddings is None:
            raise ValueError("Chưa có embeddings. Hãy gọi create_embeddings() trước.")

        documents = self.processed_documents if self.processed_documents else self.documents

        # 1. Build FAISS semantic index
        self.logger.info("Đang xây dựng FAISS semantic index...")
        dimension = embeddings.shape[1]
        self.semantic_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.semantic_index.add(embeddings.astype('float32'))

        # 2. Build BM25 keyword index
        self.logger.info("Đang xây dựng BM25 keyword index...")
        processed_texts = []
        for doc in documents:
            text = doc.get('processed_for_search', doc['content'])
            # Tokenize cho BM25
            tokens = text.lower().split()
            processed_texts.append(tokens)

        self.bm25_index = BM25Okapi(processed_texts)

        # 3. Build TF-IDF index cho backup
        self.logger.info("Đang xây dựng TF-IDF index...")
        texts_for_tfidf = [doc.get('processed_for_search', doc['content']) for doc in documents]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=list(self.vietnamese_stopwords) if self.vietnamese_stopwords else None
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts_for_tfidf)

        self.logger.info("Hoàn thành xây dựng tất cả search indices")

    def search_semantic(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Tìm kiếm semantic sử dụng embeddings
        """
        if self.semantic_index is None:
            raise ValueError("Chưa build semantic index")

        # Process query giống như documents
        processed_query = self._process_for_search(query)
        query_embedding = self.embedding_model.encode([processed_query], normalize_embeddings=True)

        # Search
        scores, indices = self.semantic_index.search(query_embedding.astype('float32'), top_k)

        documents = self.processed_documents if self.processed_documents else self.documents
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if 0 <= idx < len(documents):
                result = {
                    'rank': i + 1,
                    'score': float(score),
                    'method': 'semantic',
                    'document': documents[idx],
                    'content': documents[idx]['content']
                }
                results.append(result)

        return results

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Tìm kiếm keyword sử dụng BM25
        """
        if self.bm25_index is None:
            raise ValueError("Chưa build BM25 index")

        # Process query
        processed_query = self._process_for_search(query)
        query_tokens = processed_query.lower().split()

        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        documents = self.processed_documents if self.processed_documents else self.documents
        results = []
        for i, idx in enumerate(top_indices):
            if 0 <= idx < len(documents) and scores[idx] > 0:
                result = {
                    'rank': i + 1,
                    'score': float(scores[idx]),
                    'method': 'bm25',
                    'document': documents[idx],
                    'content': documents[idx]['content']
                }
                results.append(result)

        return results

    def search_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.6) -> List[Dict]:
        """
        Tìm kiếm hybrid kết hợp semantic và keyword

        Args:
            query: Câu hỏi tìm kiếm
            top_k: Số kết quả trả về
            alpha: Trọng số cho semantic search (0-1), (1-alpha) cho keyword search
        """
        # Lấy kết quả từ cả hai phương pháp với top_k lớn hơn
        expand_k = min(top_k * 3, 50)

        semantic_results = self.search_semantic(query, expand_k)
        bm25_results = self.search_bm25(query, expand_k)

        # Normalize scores
        if semantic_results:
            max_semantic = max(r['score'] for r in semantic_results)
            min_semantic = min(r['score'] for r in semantic_results)
            if max_semantic > min_semantic:
                for r in semantic_results:
                    r['normalized_score'] = (r['score'] - min_semantic) / (max_semantic - min_semantic)
            else:
                for r in semantic_results:
                    r['normalized_score'] = 1.0

        if bm25_results:
            max_bm25 = max(r['score'] for r in bm25_results)
            min_bm25 = min(r['score'] for r in bm25_results) if bm25_results else 0
            if max_bm25 > min_bm25:
                for r in bm25_results:
                    r['normalized_score'] = (r['score'] - min_bm25) / (max_bm25 - min_bm25)
            else:
                for r in bm25_results:
                    r['normalized_score'] = 1.0 if r['score'] > 0 else 0.0

        # Combine scores
        combined_scores = {}

        # Add semantic scores
        for result in semantic_results:
            doc_id = result['document']['id']
            combined_scores[doc_id] = {
                'semantic_score': result['normalized_score'],
                'bm25_score': 0.0,
                'document': result['document'],
                'semantic_rank': result['rank']
            }

        # Add BM25 scores
        for result in bm25_results:
            doc_id = result['document']['id']
            if doc_id in combined_scores:
                combined_scores[doc_id]['bm25_score'] = result['normalized_score']
                combined_scores[doc_id]['bm25_rank'] = result['rank']
            else:
                combined_scores[doc_id] = {
                    'semantic_score': 0.0,
                    'bm25_score': result['normalized_score'],
                    'document': result['document'],
                    'bm25_rank': result['rank']
                }

        # Calculate final scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            final_score = alpha * scores['semantic_score'] + (1 - alpha) * scores['bm25_score']

            # Bonus cho documents có cả semantic và keyword match
            if scores['semantic_score'] > 0 and scores['bm25_score'] > 0:
                final_score *= 1.2  # Boost 20%

            final_results.append({
                'document': scores['document'],
                'content': scores['document']['content'],
                'final_score': final_score,
                'semantic_score': scores['semantic_score'],
                'bm25_score': scores['bm25_score'],
                'method': 'hybrid'
            })

        # Sort by final score
        final_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Add ranks
        for i, result in enumerate(final_results[:top_k]):
            result['rank'] = i + 1

        return final_results[:top_k]

    def search(self, query: str, top_k: int = 5, method: str = 'hybrid') -> List[Dict]:
        """
        Main search function
        """
        if method == 'semantic':
            return self.search_semantic(query, top_k)
        elif method == 'bm25':
            return self.search_bm25(query, top_k)
        elif method == 'hybrid':
            return self.search_hybrid(query, top_k)
        else:
            raise ValueError("Method phải là 'semantic', 'bm25', hoặc 'hybrid'")

    def save_index(self, base_path: str):
        """
        Lưu tất cả indices và data
        """
        import os
        os.makedirs(base_path, exist_ok=True)

        # Save FAISS index
        if self.semantic_index:
            faiss.write_index(self.semantic_index, f"{base_path}/semantic_index.bin")

        # Save documents and other data
        data_to_save = {
            'documents': self.documents,
            'processed_documents': self.processed_documents,
            'document_embeddings': self.document_embeddings,
            'bm25_index': self.bm25_index,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
        }

        with open(f"{base_path}/retrieval_data.pkl", 'wb') as f:
            pickle.dump(data_to_save, f)

        self.logger.info(f"Đã lưu tất cả data tại: {base_path}")

    def load_index(self, base_path: str):
        """
        Load tất cả indices và data
        """
        # Load FAISS index
        semantic_index_path = f"{base_path}/semantic_index.bin"
        if os.path.exists(semantic_index_path):
            self.semantic_index = faiss.read_index(semantic_index_path)

        # Load other data
        with open(f"{base_path}/retrieval_data.pkl", 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.processed_documents = data['processed_documents']
        self.document_embeddings = data['document_embeddings']
        self.bm25_index = data['bm25_index']
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.tfidf_matrix = data['tfidf_matrix']

        self.logger.info(f"Đã load tất cả data từ: {base_path}")