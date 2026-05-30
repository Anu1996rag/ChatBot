import streamlit as st
from langchain_core.messages import HumanMessage
from backend import chatbot
import utils

# ****************************** Session Setup ******************************
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = utils.generate_thread_id()


def reset_chat():
    st.session_state["thread_id"] = utils.generate_thread_id()
    st.session_state["message_history"] = []


# ****************************** Sidebar UI ******************************
st.sidebar.title("Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversations")

st.sidebar.text(st.session_state["thread_id"])

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask your query here")
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state["thread_id"]}}
    with st.chat_message('assistant'):
        ai_response = st.write_stream(
            message.content for message, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})