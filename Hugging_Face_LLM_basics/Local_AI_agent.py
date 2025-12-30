from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
model = OllamaLLM(model="llama3.2")
template = """  
You are an expert in answering questions about a pizza resturaunt.

Here are the relevant reviews: {review}

Here is the question to answer: {question}
"""

# Use from_template instead of passing the string to the constructor
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
    print("\n\n--------------------------------------")
    question = input("Ask any question (q to quit):")
    if(question.lower() == 'q'):
        break
    print("\n\n")
    reviews = retriever.invoke(question)
    result = chain.invoke({"review" : reviews, "question": "What is the best pizza place in town?"})
    print(result)