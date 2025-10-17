from modules.rag_pipeline import CTUChatbot

def test_answer_query(query: str):
    bot = CTUChatbot()
    
    answer = bot.answer_query(query)
    print(f"Câu hỏi: {query}\nTrả lời: {answer}\n")

if __name__ == "__main__":
    test_answer_query()
