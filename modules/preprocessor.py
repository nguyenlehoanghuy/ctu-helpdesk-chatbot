from modules.nlp_utils import clean_text

def preprocess_documents(docs):
    return [clean_text(doc) for doc in docs]
