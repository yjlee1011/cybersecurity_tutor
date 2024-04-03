"""
Date: 3/20/2024
Purpose: To examine the return values from `answer_question()` in `cybersecurity_tutor.py`

Result

1. The returned objects only contains `quesiton` and `answer`. It does not contain `context` chunks.
2. LangSmith shows that the context chunk includes only 6 examples of cybersecurity threats. 
It may suggest that I need to change the chunk size. 
3. Even though I doubled the chunk size, the context only contains 6 examples. Why?
"""

import pandas as pd

from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory

from dotenv import load_dotenv

load_dotenv()

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
        embedding=embedding,
        persist_directory=persist_directory
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 1}
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
    return result["answer"]


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30
markdown_files = ["../030624/Video_1_final.md"]
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

questions = []
llm_answers = []

question = "How can cybersecurity threats be classified?"

llm_ans = answer_question(tutor, question)
print(llm_ans)