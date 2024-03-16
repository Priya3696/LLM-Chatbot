import streamlit as st
import random
import time
from main import run_server
from multimodal import run
from main1 import run_csv


# Streamed response emulator
def response_generator(question):
    if "@pdf" in question or "@csv" in question:
        response = run(question)
        print(f"response in streamlit: {response}")
        yield response
    elif "@json" in question:
        response = run_server(question)
        print(f"response in streamlit: {response}")
        yield response
    else:
        response = run_server(question)
        print(f"response in streamlit: {response}")
        yield response



st.title("chatbot for documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))

    # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# to run the app:
# streamlit run /Users/priya/Bot_Builder/ChatBot/stream.py