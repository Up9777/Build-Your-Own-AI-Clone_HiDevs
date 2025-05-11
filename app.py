import sys
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Example use
print("Using Groq API key:", GROQ_API_KEY[:8] + "..." if GROQ_API_KEY else "Not set")


# Fallback if using Streamlit Cloud: fix for SQLite error
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    st.warning("pysqlite3 not found; sqlite3 may cause issues on some systems.")

# --- Custom UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; padding: 20px; }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 10px;
        font-size: 16px;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 12px;
        border-radius: 15px;
        margin: 5px 10px;
        max-width: 70%;
        word-wrap: break-word;
    }
    .ai-message {
        background-color: #e9ecef;
        color: #333;
        padding: 12px;
        border-radius: 15px;
        margin: 5px 10px;
        max-width: 70%;
        word-wrap: break-word;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    h1, h3 { color: #343a40; }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

# --- Model & Embeddings Setup ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Use Streamlit secrets for Groq API Key
GROQ_API_KEY = "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"




chat = ChatGroq(
    temperature=0.7,
    model_name="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY
)

# --- ChromaDB Setup ---
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")
except Exception as e:
    st.error(f"ChromaDB Initialization Error: {e}")
    st.stop()

# --- LLM Response ---
def query_llama3(user_query):
    system_prompt = "System Prompt: Your AI clone personality based on Utkarsh Patil."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    try:
        response = chat.invoke(messages)
        st.session_state.memory.append({"input": user_query, "output": response.content, "id": str(uuid.uuid4())})
        return response.content
    except Exception as e:
        return f"⚠ API Error: {str(e)}"

# --- Streamlit App ---
def main():
    st.sidebar.markdown("### About")
    st.sidebar.write("AI Chatbot based on Utkarsh Patil, powered by Groq & Streamlit.")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.memory = []
        st.rerun()

    st.title("🤖 AI Chatbot")
    st.markdown("Welcome to your personal AI assistant. Ask anything!")

    if "memory" not in st.session_state:
        st.session_state.memory = []

    # Display previous conversation
    st.markdown("### Conversation")
    with st.container():
        for chat_entry in st.session_state.memory:
            st.markdown(
                f"""<div style='display: flex; justify-content: flex-end;'>
                        <div class='user-message'><strong>You:</strong> {chat_entry['input']}</div>
                   </div>""", unsafe_allow_html=True)
            st.markdown(
                f"""<div style='display: flex; justify-content: flex-start;'>
                        <div class='ai-message'><strong>AI:</strong> {chat_entry['output']}</div>
                   </div>""", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("Your question:", placeholder="Type your message here...")
        submit = st.form_submit_button("Send")
        if submit and user_query:
            _ = query_llama3(user_query)
            st.rerun()

if __name__ == "__main__":
    main()
