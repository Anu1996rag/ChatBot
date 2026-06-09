import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import utils
from llm_backend import chatbot, get_threads

def reset_chat():
    st.session_state["thread_id"] = utils.generate_thread_id()
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].add(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}}).values["messages"]

# ****************************** Session Setup ******************************
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = utils.generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_threads()

add_thread(st.session_state["thread_id"])

# ****************************** Sidebar UI ******************************
st.sidebar.title("Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversations")

for each_thread_id in reversed(list(st.session_state["chat_threads"])):
    if st.sidebar.button(str(each_thread_id)):
        st.session_state["thread_id"] = each_thread_id
        messages = load_conversation(each_thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({"role": role, "content": message.content})

        st.session_state["message_history"] = temp_messages


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

        def only_stream_ai_messages():
            for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_response = st.write_stream(only_stream_ai_messages())

    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})