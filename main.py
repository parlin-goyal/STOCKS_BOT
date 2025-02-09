import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url and url.strip():
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_index"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    try:
        if not urls:
            st.error("Please enter at least one valid URL")
        else:
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Text Splitter...Started...✅✅✅")
            docs = text_splitter.split_documents(data)
            
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            time.sleep(2)

            # Save FAISS index directly
            vectorstore_openai.save_local(file_path)
            st.success("Processing completed successfully!")
            
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")

query = st.text_input("Question: ")

if query:
    try:
        if os.path.exists(file_path):
            # Load FAISS index with allow_dangerous_deserialization
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])
            
            # Optionally display sources
            if "sources" in result:
                st.subheader("Sources:")
                st.write(result["sources"])
        else:
            st.warning("Please process some URLs first!")
            
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")