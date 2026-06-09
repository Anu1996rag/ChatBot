import os
import sqlite3

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
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

## Tools

search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operator: str):
    """
    Perform basic arithmetic calculations on given two numbers.
    Supported Operations: add, sub, mul, div, mod
    """
    try:
        if operator == "add":
            result = first_num + second_num
        elif operator == "sub":
            result =first_num - second_num
        elif operator == "mul":
            result =first_num * second_num
        elif operator == "div":
            if second_num == 0:
                raise {"error": "Division by zero is not allowed"}
            result =first_num / second_num
        elif operator == "mod":
            result = first_num % second_num
        else:
            return {"error": f"Unsupported operation {operator}"}

        return {"first_num": first_num, "second_num": second_num, "operation": operator, "result": result}
    except Exception as ex:
        return {"error": str(ex)}

tools = [search_tool, calculator]

llm_with_tools = llm.bind_tools(tools)

# chat node logic
def chat_node(state: ChatState):
    # step 1 : take query
    messages = state["messages"]

    # step 2 : send to llm
    response = llm_with_tools.invoke(messages)

    # step 3 : store response into state
    return {"messages": [response]}

# tool node
tool_node = ToolNode(tools)

# create sqlite database
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
# create a checkpointer
chat_bot_checkpointer = SqliteSaver(conn)

# create graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)

# add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=chat_bot_checkpointer)

def get_threads():
    unique_threads = set()
    for checkpoint in chat_bot_checkpointer.list(None):
        unique_threads.add(checkpoint.config["configurable"]["thread_id"])
    return unique_threads