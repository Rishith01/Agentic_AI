from typing import TypedDict
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import re
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="llama-3.3-70b-versatile")

class AgentState(TypedDict):
    message: str
    restructured_question: str
    answer: str

def question_analyzer(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="""Output ONLY the clarified question. 
        Strictly NO conversational filler, NO preambles, and NO extra newlines."""
    )
    response = model.invoke([system_prompt, HumanMessage(content=state['message'])])
    
    # Remove all leading/trailing whitespace and replace internal multiple newlines with a single space
    clean_content = re.sub(r'\n+', ' ', response.content).strip()
    
    state['restructured_question'] = clean_content
    print(f"Original Question: {state['message']}")
    print(f"Restructured Question: {state['restructured_question']}")
    return state

def answer_generator(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="Answer the question directly. No extra spacing."
    )
    response = model.invoke([system_prompt, HumanMessage(content=state['restructured_question'])])
    
    # Apply the same regex cleaning
    clean_answer = re.sub(r'\n+', ' ', response.content).strip()
    
    state['answer'] = clean_answer
    print(f"Final Answer: {state['answer']}")
    return state

graph = StateGraph(AgentState)
graph.add_node("question_analyzer", question_analyzer)
graph.add_node("answer_generator", answer_generator)
graph.add_edge(START, "question_analyzer")
graph.add_edge("question_analyzer", "answer_generator")
graph.add_edge("answer_generator", END)
app = graph.compile()

user_input = "How do magnets work?"
response = app.invoke({"message": user_input})