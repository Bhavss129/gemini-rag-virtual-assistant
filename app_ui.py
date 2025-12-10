# app_ui.py
import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="RAG Virtual Assistant", page_icon="🤖")

st.title("🤖 RAG Virtual Assistant")
st.write("Ask anything from your documents.")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
query = st.chat_input("Ask your question...")

if query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Show user message
    with st.chat_message("user"):
        st.write(query)

    # Call FastAPI
    try:
        res = requests.post(
            FASTAPI_URL,
            json={"question": query},
            timeout=30
        )

        if res.status_code == 200:
            answer = res.json()["answer"]
        else:
            answer = f"Error from server: {res.text}"

    except Exception as e:
        answer = f"Connection error: {e}"

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(answer)
