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
````

-----

## 🔧 Configuration Details

  * **Chunking Strategy:** `RecursiveCharacterTextSplitter` with a **chunk size of 800 tokens** and a **chunk overlap of 100 tokens**.
  * **Embedding Model:** `all-MiniLM-L6-v2` (384 dimensions).
  * **Retriever:** Pinecone vector search configured to retrieve the **top 10** most similar documents (`k=10`).
  * **Reranker:** Cohere's `rerank-english-v3.0` model, which filters the retrieved documents down to the **top 4** most relevant ones (`top_n=4`).

-----

## ✅ Minimal Evaluation (Gold Set)

A small set of questions was used to test the RAG pipeline's effectiveness.

| Question | Expected Answer Snippet | Actual Result |
| :--- | :--- | :--- |
| What is the Transformer model? | A model architecture introduced by Google in 2017. | **Success.** Provided the correct definition with citation. |
| Who is the leader in GPUs? | NVIDIA is the leader. | **Success.** Correctly identified NVIDIA. |
| What is Cohere? | A Canadian startup providing LLMs for enterprise use. | **Success.** Correctly identified Cohere's purpose. |
| What does Gemini come in? | Different sizes, such as Ultra, Pro, and Nano. | **Success.** Listed the correct sizes. |
| What is the best recipe for pasta? | An "I cannot answer" response. | **Success.** Correctly stated the answer was not in the context. |

**Success Rate:** The model achieved a **100% success rate** on this test set, correctly answering all in-domain questions with proper grounding and correctly identifying the out-of-domain question.

-----

## 💡 Remarks & Tradeoffs

  * **Model Deprecation:** During development, we encountered multiple `model_decommissioned` errors from the Groq API. This highlights a key tradeoff in using cutting-edge AI services: developers must be prepared to update their code frequently as model availability changes. We adapted by updating the `model_name` multiple times to land on a stable, supported model.
  * **Free Tier Limits:** This application was built using the free tiers of all cloud services. A production application would require upgrading to paid plans to handle higher usage and ensure availability.

-----

## 🔗 Resume Link

https://drive.google.com/file/d/1FtK7W76VybwmhcQ2PAR_4fjRuSGnVAep/view?usp=sharing

```
```
