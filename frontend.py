import streamlit as st
import utils
import queue

from mcp_backend import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage



def reset_chat():
    st.session_state["thread_id"] = utils.generate_thread_id()
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

# ****************************** Session Setup ******************************
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = utils.generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ****************************** Sidebar UI ******************************
st.sidebar.title("MCP Chatbot")

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
        # use mutable holder so that the generator can modify it
        status_holder = {"box": None}

        def only_stream_ai_messages():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:

                    for message_chunk, metadata in chatbot.stream(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=CONFIG,
                            stream_mode="messages"
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

                submit_async_task(run_stream())

                while True:
                    item = event_queue.get()
                    if item is None:
                        break

                    message_chunk, metadata = item
                    if message_chunk == "error":
                        raise metadata

                    # lazily create and update the SAME status Container when any of the tools run
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …",
                                expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True
                            )

                    # Stream only AI responses
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

        ai_response = st.write_stream(only_stream_ai_messages())

        # Finalize message when tool is actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )

    # save all the AI responses
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})