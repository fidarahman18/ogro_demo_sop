__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="OGRO SOP Oracle", page_icon="🧠")
st.title("🧠 OGRO: The Company Brain")

# --- 1. SIDEBAR: KEY & DOCUMENTS ---
with st.sidebar:
    st.header("🔐 Authentication")
    # THE FIX: Ask for the key right here. No secrets file needed.
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.markdown("---")
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Policy PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and api_key: # Only run if key exists
        if st.button("Process Documents"):
            with st.spinner("Reading & Indexing..."):
                docs = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        loader = PyPDFLoader(temp_path)
                        docs.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # Pass the key explicitly
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=OpenAIEmbeddings(openai_api_key=api_key),
                    persist_directory="./chroma_db"
                )
                st.session_state.vectorstore = vectorstore
                st.success(f"Indexed {len(splits)} chunks!")
    elif uploaded_files and not api_key:
        st.warning("⚠️ Please enter your API Key above first!")

# --- 2. MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about travel allowance, safety protocols, etc."):
    if not api_key:
        st.error("🔒 Please enter an API Key in the sidebar to chat.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        # Pass the key explicitly
        llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
        
        system_prompt = (
            "You are the 'SOP Oracle'. "
            "Use the provided context to answer. "
            "ALWAYS cite the Source document and Page number.\n\n"
            "{context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        retriever = st.session_state.vectorstore.as_retriever()
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        with st.chat_message("assistant"):
            try:
                response = rag_chain.invoke({"input": prompt})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Please upload a PDF in the sidebar to start!")
