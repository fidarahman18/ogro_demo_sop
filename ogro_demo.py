__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st 
import os
import tempfile
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API Key
load_dotenv()

# --- CONFIGURATION ---
st.set_page_config(page_title="OGRO SOP Oracle", page_icon="🧠")
st.title("🧠 OGRO: The Company Brain")

# --- 1. SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Company Policy PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Reading & Indexing..."):
                docs = []
                # Create a temporary directory to handle files
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        loader = PyPDFLoader(temp_path)
                        docs.extend(loader.load())

                # Chunking
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # Vector Database
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=OpenAIEmbeddings(),
                    persist_directory="./chroma_db"
                )
                st.session_state.vectorstore = vectorstore
                st.success(f"Indexed {len(splits)} chunks!")

# --- 2. MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about travel allowance, safety protocols, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectorstore" in st.session_state:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        system_prompt = (
            "You are the 'SOP Oracle' for OGRO. "
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
            response = rag_chain.invoke({"input": prompt})
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    else:
        st.error("⚠️ Please upload documents first!")
