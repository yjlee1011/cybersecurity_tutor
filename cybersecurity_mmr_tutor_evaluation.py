"""
Date: 4/2/2024

Objective: Evaluate cybersecurity_mmr_tutor.py
Input: evaluation_questions.txt
Output: evaluation_questions_answers_040224.csv
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
        #documents=md_header_splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    # Create a retriever
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 4}
    )

    # retriever = vectordb.as_retriever(
    #     search_kwargs={"k": 2}
    # )

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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 30
markdown_files = ["../030624/Video_1_final_yjlee.md"]
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

with open("evaluation_questions.txt", "r") as f:
    for line in f.readlines():
        question = line.strip()
        questions.append(question)


for question in questions:
    llm_ans = answer_question(tutor, question)
    llm_answers.append(llm_ans)

df = pd.DataFrame(data={
    "question": questions,
    "answer": llm_answers
})

df.to_csv("evaluation_questions_answers_040224.csv", index=False)