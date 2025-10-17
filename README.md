# CTU Helpdesk Chatbot - Retrieval-Augmented Generation (RAG) Pipeline

## Project Overview

This project is a Master's Thesis work focused on building an AI-powered chatbot for the CTU Helpdesk system using a Retrieval-Augmented Generation (RAG) approach. The chatbot combines document retrieval techniques with a large language model (LLM) to provide accurate and concise answers to user queries based on internal CTU helpdesk documentation.

## System Architecture

- Retriever: Uses a hybrid retrieval system combining FAISS vector search with traditional BM25 and TF-IDF methods to find the most relevant documents.
- Generator: Employs a fine-tuned large language model (AutoModelForCausalLM) from Hugging Face Transformers to generate natural language answers based on retrieved context.
- Pipeline: Integrates the retriever and generator into a unified RAG pipeline for efficient and effective question answering.

## Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- FAISS (CPU or GPU version)
- Additional dependencies: tqdm, numpy, scikit-learn, etc.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nguyenlehoanghuy/ctu_helpdesk_chatbot.git
cd ctu_helpdesk_chatbot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the model and retrieval index:
   - Place the fine-tuned LLM in the models/ directory or specify the correct path.
   - Place the pre-built retrieval index in the data/ directory or specify the correct path.

## Notes

- The model loads with dtype=torch.bfloat16 if GPU with support is available, otherwise defaults to float32.
- The chatbot uses a chat prompt template (apply_chat_template) to format input for the language model.
- Ensure the retrieval index and model paths are correctly specified before running.

## License & Acknowledgments

This project is part of my Master's project in Information Systems at Can Tho University. All rights reserved.
