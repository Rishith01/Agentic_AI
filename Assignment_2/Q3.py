from typing import Sequence, TypedDict, Annotated
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

model = ChatGroq(model="llama-3.3-70b-versatile")

def python_agent(state: AgentState):
    system_prompt = SystemMessage(content="You are specialized in Python programming. Answer concisely and clearly.")
    response = model.invoke([system_prompt] + list(state['messages']))
    return {"messages": [response]}

def general_qa_agent(state: AgentState):
    system_prompt = SystemMessage(content="You are a general QA assistant.")
    response = model.invoke([system_prompt] + list(state['messages']))
    return {"messages": [response]}

def router(state: AgentState):
    last_message = state['messages'][-1].content.lower()
    if "python" in last_message or "code" in last_message:
        return "python"
    return "general"

graph = StateGraph(AgentState)
graph.add_node("python_node", python_agent)
graph.add_node("general_node", general_qa_agent)

graph.add_conditional_edges(
    START,
    router,
    {
        "python": "python_node",
        "general": "general_node"
    }
)

graph.add_edge("python_node", END)
graph.add_edge("general_node", END)

app = graph.compile()

questions = [
    "Write a python script to print hello world",
    "Tell me some locations to visit in the summer vacations of India."
]

for q in questions:
    print(f"\n--- Input Question: {q} ---")
    inputs = {"messages": [HumanMessage(content=q)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Agent ({key}): {value['messages'][-1].content}")