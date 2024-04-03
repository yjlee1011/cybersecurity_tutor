Date: 4/3/2024

# data/Video_1_final_yjlee.md

1. Copied the file from ../030624/
2. Use git to serve Streamlit Cloud app
3. To get around sqlite3 chroma error, manually add chromadb and pysqlite3-binary to requirements.txt
4. To get around sqlite3 chroma error, modify cybersecurity_mmr_tutor.py (see https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq)

Date: 4/2/2024

# cybersecurity_mmr_tutor_evaluation.py

Evaluate the mmr-based tutor using evaluation_questions.txt => evaluation_questions_answers_040224.csv

Date: 4/1/2024

# Video_1_final_yjlee.md

Since markdown splitter does not use the header text, I manually copied the header text to the body.

# cybersecurity_mmr_tutor

I used MarkdownHeaderTextSplitter and RecursiveCharacterSplitter, but IAAA does not work.
Also, the chunk does not include the copied header title. Why?

# Video_1_final_yjlee_040124.md

Even though I copied the header text to the body at the top of the paragraph as a standalone line, the retrieved chunk does not seem to include the copied text. 
In this file, the header text is copied as the part, not a standalone line, of the main text.

Date: 3/31/2024

# cybersecurity_tutor.py fail to answer "What is IAAA?"

smith.langchain.com shows that the retriever retrieved two identical chunks, Information assurance can be done with the three important points like confidentiality, integrity, and availability. Simply this is termed as CIA.

Since cybersecurity_tutor.py uses the default similarity search, I will see if using mmr can resolve this issue in cybersecurity_mmr_tutor.py

# cybersecurity_mmr_tutor.py

After deleting the chorma db, the MMR-based retrieved correctly answered the "What is IAAA?" question. 
See test_033124.txt for detail.