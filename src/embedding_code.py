from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import sentence_transformers
import os
import glob
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_embeddings(chunks_dir,vectordb_dir):


    chunk_files = glob.glob(os.path.join(chunks_dir, "chunk_*.txt"))
    documents = []

    for chunk_file in chunk_files:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Create Document object
            doc = Document(page_content=content, metadata={"source": chunk_file})
            documents.append(doc)


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vectordb directory
    os.makedirs(vectordb_dir, exist_ok=True)

    # Create vector database
    print("Creating embeddings and storing in FAISS vector database...")
    vectordb = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Save the database
    vectordb.save_local(vectordb_dir)
    print(vectordb)

    return vectordb



