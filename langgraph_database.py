import os
import sqlite3

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    max_tokens=500
)

# create state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# chat node logic
def chat_node(state: ChatState):
    # step 1 : take query
    messages = state["messages"]

    # step 2 : send to llm
    response = llm.invoke(messages)

    # step 3 : store response into state
    return {"messages": [response]}

# create sqlite database
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
# create a checkpointer
chat_bot_checkpointer = SqliteSaver(conn)

# create graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=chat_bot_checkpointer)

def get_threads():
    unique_threads = set()
    for checkpoint in chat_bot_checkpointer.list(None):
        unique_threads.add(checkpoint.config["configurable"]["thread_id"])
    return unique_threads