from typing import TypedDict, Annotated, Sequence
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

model = ChatGroq(model="llama-3.3-70b-versatile")

def qa_answering(state: AgentState):
    system_prompt = SystemMessage(
        content="You are an expert in answering questions pertaining to maths, coding or general knowledge."
    )
    # Include system prompt in the call
    response = model.invoke([system_prompt] + list(state['messages']))
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("process", qa_answering)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Testing with 3 specific queries
test_queries = [
    "Coding: Write a Python function to reverse a string.",
    "Math: What is the derivative of x^2?",
    "General Knowledge: Who painted the Mona Lisa?"
]

current_state = {"messages": []}

for query in test_queries:
    print(f"\nUser: {query}")
    current_state = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(f"AI: {current_state['messages'][-1].content}")

with open("Q1_history.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for message in current_state['messages']:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("\nConversation saved to Q1_history.txt")