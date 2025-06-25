import streamlit as st
from src.ret_gen import get_final_output, get_model_info, get_chunk_count

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
user_query = st.text_input("Enter your query")

if st.button("Ask"):
    if uploaded_file is None or not user_query:
        st.warning("Please upload a PDF and enter a query.")
    else:
        with st.spinner("Processing..."):
            response, source_docs = get_final_output(user_query, uploaded_file)
            
        st.success("Answer:")
        st.write(response)
        
        # Expandable source documents
        with st.expander("View Source Documents"):
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Source {i+1}:**")
                st.text_area(f"Document {i+1}", doc.page_content, height=150, key=f"doc_{i}")
                st.markdown("---")
        
        # Update chunk count after processing
        chunk_count = len(list(source_docs))
        
        # Sidebar
        with st.sidebar:
            st.header("System Info")
            st.write(f"**Model:** {get_model_info()}")
            st.write(f"**Chunks:** {chunk_count}")
else:
    # Default sidebar when no processing
    with st.sidebar:
        st.header("System Info")
        st.write(f"**Model:** {get_model_info()}")
        st.write("**Chunks:** 0")