# Mini RAG: AI Engineer Assessment

This project is a fully functional Retrieval-Augmented Generation (RAG) application built with Streamlit and LangChain. It ingests a document, stores its content in a hosted vector database, and uses a reranker and an LLM to answer questions based on that document, complete with source citations.

---

## 🏛️ Architecture

The application follows a classic RAG pipeline:

1.  **Ingestion:** A source text document is loaded, split into smaller chunks, and converted into vector embeddings using a Hugging Face model.
2.  **Storage:** The text chunks and their corresponding embeddings are stored in a hosted **Pinecone** vector database.
3.  **Retrieval:** When a user asks a question, the application retrieves the most relevant document chunks from Pinecone based on semantic similarity.
4.  **Reranking:** The retrieved chunks are re-scored for relevance by **Cohere's Rerank** model to improve the quality of the context.
5.  **Generation:** The top-ranked chunks and the user's original question are sent to a **Groq** LLM, which generates a final answer grounded in the provided sources.



---

## ⚙️ Tech Stack & Providers

* **Application Framework:** Streamlit
* **Core Logic:** LangChain
* **Vector Database:** Pinecone
* **Embedding Model:** Hugging Face (`all-MiniLM-L6-v2`)
* **Reranker:** Cohere (`rerank-english-v3.0`)
* **LLM Provider:** Groq (`llama-3.1-8b-instant`)

---

## 🚀 Setup & Running

### 1. Prerequisites

* Python 3.9+
* Accounts for [Pinecone](https://www.pinecone.io/), [Cohere](https://cohere.com/), and [Groq](https://console.groq.com/keys).

### 2. Installation

Clone the repository, install the dependencies, and configure your API keys.

```bash
# Clone this repository
git clone <YOUR_REPOSITORY_LINK>
cd mini-rag-project

# Install required packages
python -m pip install -r requirements.txt

# Create your secrets file by copying the example
# On Windows:
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# On macOS/Linux:
# cp .streamlit/secrets.toml.example .streamlit/secrets.toml