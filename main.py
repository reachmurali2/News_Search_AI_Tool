__import__('pysqlite3')                                                 # Dynamically imports pysqlite3 (a standalone SQLite module that supports modern features).
import sys                                                               
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')                   # Replaces the default sqlite3 module with pysqlite3. This is helpful in environments like Streamlit Cloud where SQLite might not support fulltext search (FTS5) or other features used in vector databases.

import streamlit as st                                                  # Loads Streamlit for building the web UI.
from rag import process_urls, generate_answer                           # Imports two custom functions:

st.title("News Search Tool")                                            # Sets the page title shown at the top of the web UI.

url1 = st.sidebar.text_input("URL 1")                                  # Displays input fields in the sidebar to enter up to 3 URLs.
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

placeholder = st.empty()                                               # Creates a placeholder element that can be updated later with text or widgets dynamically. Used to show messages or the question box.

api_key = st.secrets["GROQ_API_KEY"]                                   # Loads the GROQ API key from .streamlit/secrets.toml or Streamlit Cloud’s app secrets securely. Required for RAG operations using the Groq API.

process_url_button = st.sidebar.button("Process URLs")                 # Adds a button in the sidebar to trigger URL processing.
if process_url_button:                                                 # If the button is clicked, gather non-empty URLs into a list.
    urls = [url for url in (url1, url2, url3) if url!='']
    if len(urls) == 0:                                                 # If no URLs were entered, show a message prompting the user to input at least one.
        placeholder.text("You must provide at least one valid url")
    else:
        for status in process_urls(urls, api_key):                     # Calls the process_urls function with the URL list and API key.
            placeholder.text(status)                                   # Each status returned is dynamically shown using the placeholder. This could be messages like "Processing...", "Done", or error info.

query = placeholder.text_input("Question")                             # Replaces the placeholder with a text input field for asking questions after URLs are processed.
if query:                                     
    try:
        answer, sources = generate_answer(query)                      # If a query is entered, call generate_answer() to perform retrieval-augmented generation (RAG). It returns the answer and its sources.
        st.header("Answer:")                                          # Displays the answer using markdown, and then prints out each source link on a new line.
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:                                        # Error Handling , If the user queries before URL processing is complete, shows a warning. You could also log the actual exception e for debugging.
        placeholder.text("You must process urls first")


