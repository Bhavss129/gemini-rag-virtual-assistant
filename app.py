# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="RAG Virtual Assistant with Gemini 2.5 Flash")

# ----- Load Vector DB -----
CHROMA_PATH = "vectordb/"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ----- Gemini LLM -----
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# ----- Build RAG pipeline -----
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ----- API Schema -----
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: Query):
    try:
        response = qa_chain.run(payload.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
