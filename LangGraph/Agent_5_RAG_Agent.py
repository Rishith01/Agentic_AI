from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# Using Llama 3 on Groq
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile", 
#     temperature=0
# )
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pdf_path = "LangGraph/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF loaded: {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = "./chroma_db"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Vector store created locally!")
except Exception as e:
    print(f"Error: {str(e)}")
    raise

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

@tool
def retriever_tool(query: str) -> str:
    """Searches the Stock Market Performance 2024 document."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    
    return "\n\n".join([f"Doc {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are a Stock Market AI. 
Use the retriever tool for data and cite the documents you use for retrieving. 
If you cannot find information after 2-3 attempts, inform the user that the document does not contain that information.
"""

tools_dict = {t.name: t for t in tools}

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])[-5:]
    message = llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Tool: {t['name']}")
        result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG AGENT (GROQ + HF) ===")
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() in ['exit', 'quit']: break
        result = rag_agent.invoke({"messages": [HumanMessage(content=user_input)]})
        print(f"\nANSWER: {result['messages'][-1].content}")

if __name__ == "__main__":
    running_agent()