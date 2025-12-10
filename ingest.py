# ingest.py
import os
import argparse
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from tqdm import tqdm

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv



load_dotenv()

DATA_PATH = "data/"
CHROMA_PATH = "vectordb/"

def load_docs(data_dir: str):
    docs = []
    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")

    for file in p.rglob("*"):
        if file.suffix.lower() in [".txt", ".md"]:
            docs.extend(TextLoader(str(file), encoding="utf-8").load())
        elif file.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())

    return docs
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    return chunks

def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    vectordb.persist()
    print("Vector DB stored at:", CHROMA_PATH)

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_docs(DATA_PATH)
    print(f"Loaded {len(docs)} docs")

    print("Chunking...")
    chunks = chunk_docs(docs)
    print(f"Created {len(chunks)} chunks")

    print("Creating vector DB...")
    create_vector_db(chunks)

    print("Ingestion complete")
