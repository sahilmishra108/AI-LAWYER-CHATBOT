# 🧠 AI-Lawyer Chatbot – Legal Question Answering using RAG & DeepSeek R1

The **AI-Lawyer Chatbot** is an intelligent legal assistant that uses **Retrieval-Augmented Generation (RAG)** to answer legal questions based on uploaded PDF documents. Powered by the **DeepSeek R1** model and the **LangChain** framework, it combines document retrieval and large language models to generate grounded and context-aware responses.

## 📌 Key Features

- Upload any legal or human rights PDF
- Context-aware legal Q&A using DeepSeek R1 via Groq
- Semantic document retrieval using FAISS and Ollama embeddings
- Built using LangChain, Streamlit, and Ollama
- Reduces hallucination and ensures context fidelity via RAG

## 📂 Project Structure
| File/Folder                              | Description                                      |
|------------------------------------------|--------------------------------------------------|
| 📁 `AI-Lawyer-Chatbot/`                  | Root project directory                           |
| ├── `main.py`                            | End-to-end chatbot application                   |
| ├── `frontend.py`                        | Minimal UI version                               |
| ├── `rag_pipeline.py`                   | LLM + Retrieval + Prompt engineering             |
| ├── `vector_database.py`                | Vector DB creation using FAISS                   |
| ├── `universal_declaration_of_human_rights.pdf` | Example legal document                    |
| ├── `requirements.txt`                  | Dependency list (for pip users)                  |
| ├── `Pipfile` / `Pipfile.lock`          | Pipenv environment specification                 |
| ├── `Readme.md`                         | Project documentation                            |
| └── `vectorstore/`                      | Folder to store FAISS vector index               |


## 🧱 Core Technologies

- **DeepSeek R1** – Language model for legal reasoning (used via Groq and Ollama)
- **LangChain** – Framework to build and manage the RAG pipeline
- **FAISS** – Efficient vector similarity search
- **Ollama** – Embedding generation using locally hosted LLMs
- **Streamlit** – For building the chatbot user interface
- **PDFPlumber** – Extracts and loads PDF text content

## 📐 Architecture

### Phase 1 – User Interface (Streamlit)

- Upload PDF
- Enter question

### Phase 2 – Vector Store Setup

- PDF loading
- Chunking (RecursiveCharacterTextSplitter)
- Embedding with DeepSeek via Ollama
- Store vectors in FAISS

### Phase 3 – RAG Inference Pipeline

- Retrieve relevant chunks from FAISS
- Answer using DeepSeek R1 with Groq
  ![Screenshot 2025-06-08 001001](https://github.com/user-attachments/assets/d23e6f18-9a2c-4afa-b57f-964a4d4463fc)


## ⚙️ Setup Guide (Quick Start)

### 🔹 Option 1: Pipenv (Recommended)

```bash
pip install pipenv
pipenv install
pipenv shell
```
🔹 Option 2: pip + virtualenv
```bash
pip install virtualenv
virtualenv venv

# Activate the environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```
🔹 Option 3: Conda

```bash
conda create -n ai-lawyer python=3.13
conda activate ai-lawyer
pip install -r requirements.txt
```

## 🚀 Run the Application

### 📦 Full Pipeline

```bash
streamlit run main.py
```
## ▶️ Run Components Individually

### 🧩 Frontend Only

```bash
streamlit run frontend.py
```
### 🗃️ Vector DB Setup

```bash
python vector_database.py
```
### 🧠 Run RAG Pipeline

```bash
python rag_pipeline.py
```
![Screenshot 2025-06-19 191440](https://github.com/user-attachments/assets/8d7a9139-1f79-4f74-9610-1ed1d1da2185)

## 🧪 Example Use Case

Upload `universal_declaration_of_human_rights.pdf` and ask:


If a government forbids the right to assemble peacefully, which articles are violated?

The chatbot will:

🔍 Search for relevant sections in the PDF

🧠 Construct a context-aware prompt

💬 Generate a grounded legal explanation using DeepSeek R1
  

📘 What is RAG?
---
Retrieval-Augmented Generation (RAG) enhances LLMs by integrating relevant external documents. It helps:

🎯 Anchor responses in real data

⚖️ Improve factual accuracy

🚫 Reduce hallucinations

RAG Process:
Vectorize and store documents

Retrieve semantically relevant content

Generate answers using retrieved context

## 📚 References

- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- [Groq](https://groq.com/)
- [DeepSeek R1](https://deepseek.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
---
  📄 License
  ---
MIT License – feel free to use, modify, and distribute with credit.


  












 
