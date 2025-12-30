import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from langchain_core.documents import Document

df = pd.read_csv("venv/realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_document = not os.path.exists(db_location)

if add_document:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content = row['Title'] + " " + row['Title'],
            metadata = {'rating' : row['Rating'], 'date': row['Date']},
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_document:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k":5})


    