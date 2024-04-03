"""
Date: 3/21/2024

Objective
1. In `cybersecurity_tutor_debug1.py`, doubling the chunk size did not include the context window size. Why?

Even though I did not use RecursiveCharacterTextSplitter to ensure that the entire text under each header will form a Document.
However, I found that it did not resolve the issue; the context still contains 6 category.
The reason is that the vectorstore (Chroma) needs to be removed before running the app. 
Otherwise, the app uses an existing vectorstore in which a Document instance contains only 6 categories. 
"""

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

markdown_files = ["../030624/Video_1_final_yjlee.md"]

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

persist_directory = PERSIST_DIR = './chroma/'

# Read a markdown file
document_txt = ""
for md_file in markdown_files:
    with open(md_file) as f:
        document_txt += f.read()


# Split the markdown text
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

md_header_splits = markdown_splitter.split_text(document_txt)

# I confirmed that one of Document instance contains 12 cyberthreat categories
# print(md_header_splits)

# Create an embedding
embedding = OpenAIEmbeddings()

# Create a database of embedding
vectordb = Chroma.from_documents(
    documents=md_header_splits,
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

question = "How can cybersecurity threats be classified?"
result = qa.invoke({"question": question})
print(result["answer"])