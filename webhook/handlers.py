from modules.rag_pipeline import handle_user_message as rag_answer

def handle_user_message(user_message: str) -> str:
    """
    Gọi pipeline RAG để sinh câu trả lời.
    """
    return rag_answer(user_message)
