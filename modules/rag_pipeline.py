import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_PATH, INDEX_PATH, LOG_FILE
from modules.retriever import RetrievalSystem
import logging
import traceback

# -----------------------
# Logging setup
# -----------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


class CTUChatbot:
    """
    Chatbot pipeline gồm 2 phần:
    - Retriever: RetrievalSystem (FAISS + BM25 + TF-IDF hybrid search)
    - Generator: LLM model (AutoModelForCausalLM)
    """
    def __init__(self, model_path=MODEL_PATH, index_path=INDEX_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing CTUChatbot (device: {self.device})")

        # --- Load retriever ---
        self.retrieval = RetrievalSystem()
        try:
            self.retrieval.load_index(index_path)
            logger.info(f"✅ Loaded retrieval index from: {index_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load retrieval index: {e}")
            raise

        # --- Load model ---
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

        logger.info("💡 CTUChatbot ready to serve.")

    def answer_query(self, query: str) -> str:
        """
        Nhận câu hỏi, retrieve context, sinh câu trả lời ngắn gọn.
        """
        try:
            if not query.strip():
                return "Xin hãy nhập câu hỏi hợp lệ."

            # Retrieve top documents
            results = self.retrieval.search(query, top_k=1, method='hybrid')
            if not results:
                logger.warning(f"⚠️ No retrieval results for query: {query}")
                context = "(Không tìm thấy dữ liệu phù hợp.)"
            else:
                # Nếu có parent_id, lấy nội dung gốc, tránh mất ngữ cảnh
                contexts = []
                for r in results:
                    doc = r.get("document", {})
                    parent_id = doc.get("parent_id", doc.get("id"))
                    full_doc = next((d for d in self.retrieval.documents if d["id"] == parent_id), None)
                    if full_doc:
                        contexts.append(full_doc.get("original_content", full_doc.get("content", "")))
                    else:
                        contexts.append(r.get("content", ""))
                context = "\n\n".join(contexts)

            # Build prompt
            prompt = f"Dựa vào thông tin sau đây về hệ thống CTU Helpdesk:\n\n{context}\n\nHãy trả lời câu hỏi: {query}"

            messages = [
                {"role": "system", "content": "Bạn là trợ lý AI cho hệ thống Helpdesk CTU."},
                {"role": "user", "content": prompt}
            ]

            # Tokenize & Generate
            # Nếu tokenizer chưa hỗ trợ apply_chat_template thì có thể tạo prompt thủ công
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback: nối chuỗi đơn giản
                text = f"{messages[0]['content']}\n{messages[1]['content']}"

            inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)

            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # Lấy phần output generated (loại bỏ phần prompt input)
            new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return answer

        except Exception as e:
            logger.error(f"❌ Error while generating answer: {e}")
            logger.debug(traceback.format_exc())
            return "Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi."


# -----------------------
# Global instance handler
# -----------------------
_chatbot_instance = None


def handle_user_message(user_message: str) -> str:
    """
    Hàm gọi nhanh (dùng cho webhook Flask Messenger)
    - Giữ instance model + retriever trong RAM (singleton)
    """
    global _chatbot_instance

    try:
        if _chatbot_instance is None:
            _chatbot_instance = CTUChatbot()
            logger.info("🌟 New chatbot instance created in memory.")

        return _chatbot_instance.answer_query(user_message)

    except Exception as e:
        logger.error(f"❌ Error in handle_user_message: {e}")
        return "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
