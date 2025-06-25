from src.chunk import create_chunk
from src.embedding_code import create_embeddings
import os
import glob
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

chunks_dir = "C:\\Users\\vansh\\Amlgo_RAG_project\\chunks"
vectordb_dir = "C:\\Users\\vansh\\Amlgo_RAG_project\\vectordb"
data_dir = "C:\\Users\\vansh\\Amlgo_RAG_project\\data"

model = "llama3.1:latest"
llm = OllamaLLM(model=model)

def get_model_info():
    return model

def get_chunk_count():
    if os.path.exists(chunks_dir):
        chunk_files = glob.glob(os.path.join(chunks_dir, "chunk_*.txt"))
        return len(chunk_files)
    return 0

def llm_part(relevant, user_query):
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, just say you don't know.
        
        {context_text}
        Question: {user_query}
        """,
        input_variables=['context_text', 'user_query']
    )
    
    context_text = "\n\n".join(doc.page_content for doc in relevant)
    
    final_prompt = prompt.invoke({"context_text": context_text, "user_query": user_query})
    answer = llm.invoke(final_prompt)
    
    return answer

def retreiver(vectordb, user_query):
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    relevant = retriever.invoke(user_query)
    return relevant

def get_final_output(user_query, uploaded_file):
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save uploaded file to data folder
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Pass file path to create_chunk instead of uploaded_file
    create_chunk(file_path)
    
    vectordb = create_embeddings(chunks_dir, vectordb_dir)
    
    relevant = retreiver(vectordb, user_query)
    
    output = llm_part(relevant, user_query)
    
    return output, relevant