
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_cohere import CohereRerank
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os
import tiktoken


try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
    PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Please add it to your .streamlit/secrets.toml file.")
    st.stop()


os.environ["COHERE_API_KEY"] = COHERE_API_KEY


def calculate_token_count(text):
    """Calculates the token count of a given text."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

@st.cache_resource
def get_models():
    """Initializes and returns the embedding model and LLM."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY, temperature=0.2)
    return embeddings, llm

embeddings, llm = get_models()


def ingest_data():
    """Loads, chunks, and stores the document in Pinecone."""
    with st.spinner("Ingesting document... This may take a moment."):
        try:
            loader = TextLoader("./data.txt")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len
            )
            split_docs = text_splitter.split_documents(docs)

            PineconeVectorStore.from_documents(
                split_docs,
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings
            )
            st.session_state.ingested = True
            st.success("Data ingestion complete!")
        except Exception as e:
            st.error(f"An error occurred during ingestion: {e}")

# --- RAG CHAIN SETUP ---

def setup_rag_chain():
    """Sets up the RAG chain with retriever, reranker, and LLM."""
    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        
        # 1. Set up the base retriever
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # 2. Set up the Cohere reranker
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=4)
        
        # 3. Create the compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )

        # 4. Create the rest of the chain
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question based *only* on the following context.
        Provide inline citations by citing the source number in square brackets [source N].
        If the answer is not in the context, state that you cannot answer the question with the provided information.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """)

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)

        return rag_chain
    except Exception as e:
        st.error(f"Failed to set up RAG chain: {e}")
        return None


st.set_page_config(page_title="Mini RAG Project", layout="centered")
st.title("📄 Mini RAG: AI Engineer Assessment")

if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


with st.sidebar:
    st.header("Controls")
    st.write("First, ingest the data into the vector database. This only needs to be done once.")
    if st.button("Ingest Data"):
        ingest_data()
    if st.session_state.ingested:
        st.success("Data is ready!")
    else:
        st.warning("Please ingest data to begin.")


if st.session_state.ingested:
    rag_chain = setup_rag_chain()

    if rag_chain and (query := st.chat_input("Ask a question about the document...")):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response["answer"]
                    st.markdown(answer)

                    context_text = " ".join([doc.page_content for doc in response["context"]])
                    token_count = calculate_token_count(context_text + query + answer)
                    st.info(f"Approximate tokens used for this response: {token_count}")

                    # Display retrieved and reranked sources
                    with st.expander("Show Sources"):
                        st.write("Top 4 relevant sources after reranking:")
                        for i, doc in enumerate(response["context"]):
                            st.info(f"**Source {i+1}**\n\n{doc.page_content}")
                            st.write(f"*Metadata:* `{doc.metadata}`")

                except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    answer = "Sorry, I encountered an error. Please try again."
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please click 'Ingest Data' in the sidebar to start the RAG pipeline.")