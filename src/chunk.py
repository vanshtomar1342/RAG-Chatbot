from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os



def create_chunk(uploaded_file):
    loader = PyPDFLoader(uploaded_file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    chunks_dir = "C:\\Users\\vansh\\Amlgo_RAG_project\\chunks"
    os.makedirs(chunks_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk_filename = os.path.join(chunks_dir, f"chunk_{i+1}.txt")
        with open(chunk_filename, 'w', encoding='utf-8') as f:
            f.write(chunk.page_content)

    return chunks