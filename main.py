__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from dotenv import load_dotenv 
from rag import process_urls, generate_answer

st.title("News Search Tool")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()

# Dual secret loading: Streamlit secrets or .env
api_key = st.secrets.get("GROQ_API_KEY")
if api_key is None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

if api_key is None:
    st.error("GROQ_API_KEY not found. Please set it in Streamlit secrets or .env file.")
    
process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url!='']
    if len(urls) == 0:
        placeholder.text("You must provide at least one valid url")
    else:
        for status in process_urls(urls, api_key):
            placeholder.text(status)

query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:
        placeholder.text("You must process urls first")
