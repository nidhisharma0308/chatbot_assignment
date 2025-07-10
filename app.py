import streamlit as st 
st.set_page_config(page_title="PDF Chatbot", page_icon="📄") 

from src.rag_utils import load_and_process, create_or_load_vectorstore, format_prompt
from llama_cpp import Llama
import os

# === PATH SETTINGS ===
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q2_K.gguf" 
DATA_PATH = "data/AI_Training_Document.pdf"  
VECTORDB_PATH = "vectordb"  

# === Load and Cache Vector Store ===
st.session_state.setdefault("vector_store", None)
if st.session_state["vector_store"] is None:
    with st.spinner("Processing PDF and building vector store..."):
        docs = load_and_process(DATA_PATH)
        vector_db = create_or_load_vectorstore(docs, VECTORDB_PATH)
        st.session_state["vector_store"] = vector_db

# === Load and Cache LLM ===
@st.cache_resource
def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_batch=128,
        use_mlock=False,
        verbose=False
    )

llm = load_llm()

# === Streamlit UI ===
st.title("📄 RAG Chatbot")
st.sidebar.title("🧠 RAG Settings")
st.sidebar.markdown("**Model:** `llama-2-7b-chat.gguf`")
st.sidebar.markdown("**Embedding:** MiniLM-L6-v2")
st.sidebar.markdown("**Chunk Size:** 250 words")
st.sidebar.markdown("**Top K Retrieval:** 5")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input & Processing
if query := st.chat_input("Ask something from the document..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Retrieve relevant chunks
    retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    print("getting relevant documnets :", docs)
    context = "\n\n".join([doc.page_content for doc in docs])
    print("getting the most relevant context : " , context)
    # Build prompt
    prompt = format_prompt(context, query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        collected_response = ""

        # Stream the response token by token
        for chunk in llm.create_completion(
            prompt=prompt,
            stream=True,
            max_tokens=512,
            temperature=0.3,
            top_p=0.95
        ):
            if "choices" in chunk:
                token = chunk["choices"][0]["text"]
                if token:
                    collected_response += token
                    message_placeholder.markdown(collected_response + "▌")

        # Final output (remove blinking cursor)
        message_placeholder.markdown(collected_response)


    # Save assistant message
    st.session_state.chat_history.append({"role": "assistant", "content": collected_response})
