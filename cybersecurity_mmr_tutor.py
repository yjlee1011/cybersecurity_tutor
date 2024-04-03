"""
Date: 3/31/2024

Extends cybersecurity_tutor.py which uses a default similarity search.
This version will use a mmr-based retriever instead to see it can fetch relevant document for "What is IAAA?"

Result:
Yes, now I got the expected outcome. 
Interestingly, the retriever did not fetch the chunk with "# What is IAAA?" title. 
It appears that the md title is not included in the chunk. 
It may suggest that I need to find a way to incorporate the md header text. 
"""

import streamlit as st

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory

import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

def create_tutor(md_files, headers_to_split_on, persist_directory):
    # Read a markdown file
    document_txt = ""
    for md_file in md_files:
        with open(md_file) as f:
            document_txt += f.read()
            #print(document_txt)

    # Split the markdown text
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(document_txt)

    # Split text in each markdown split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(md_header_splits)

    # Create an embedding
    embedding = OpenAIEmbeddings()

    # Create a database of embedding
    vectordb = Chroma.from_documents(
        documents=splits,
        #documents=md_header_splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    # Create a retriever
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 4}
    )

    # Create an OpenAI chat instance
    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(
        model_name=llm_name,
        temperature=0
    )

    # Create a memory to keep the context
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10
    )

    # Create a tutor utilizing ChatGPT
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    return qa

def answer_question(qa, question):
    result = qa.invoke({"question": question})
    return result['answer']

st.title("Ask anything about Cybersecurity!!!")

button_css = """.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""

st.markdown(f'<style>{button_css}</style>', unsafe_allow_html=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30
markdown_files = ["data/Video_1_final_yjlee.md"]
PERSIST_DIR = './chroma/'

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

tutor = create_tutor(
    markdown_files,
    headers_to_split_on,
    PERSIST_DIR
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            ai_response = answer_question(tutor, prompt)
            message_placeholder.markdown(ai_response)
            new_ai_message = {"role": "assistant", "content": ai_response}
            st.session_state.messages.append(new_ai_message)