import os
from uuid import uuid4                                                 # Generates unique IDs for each document chunk.
from dotenv import load_dotenv                                         # Loads environment variables (such as API keys) from a .env file.
from pathlib import Path                                               # Used to manage file system paths (OS independent).
from langchain.chains import RetrievalQAWithSourcesChain               # Creates a retrieval-augmented question-answering (RAG) chain that can also provide source documents.
from langchain_community.document_loaders import UnstructuredURLLoader # Loads and parses web pages from URLs into raw text.
from langchain.text_splitter import RecursiveCharacterTextSplitter     # Splits long text documents into smaller, manageable chunks based on separators.
from langchain_chroma import Chroma                                    # Vector database used to store and retrieve embeddings
from langchain_huggingface import HuggingFaceEmbeddings                # Loads embedding models from Hugging Face for converting text to vectors.
from langchain_groq import ChatGroq                                    # Loads Groq-hosted Llama models to generate answers

# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings

# from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")                                      # import api_key from .env file

# Constants
CHUNK_SIZE = 1000                                                      # Max tokens/characters per document chunk
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"             # Pre-trained embedding model
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"      # Path to store Chroma database
COLLECTION_NAME = "real_estate"                                        # Name of the vector database collection

llm = None
vector_store = None                                                    # Placeholders for the language model and vector store.


def initialize_components(api_key):                                     # Initializes: Groq’s Llama model, Hugging Face embedding model, Chroma vector database
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500, api_key=api_key)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}          # trust_remote_code=True allows the use of custom model loading logic (can be replaced with safer alternatives if needed).
        )

        vector_store = Chroma(                                # Initializes a Chroma vector store object. This is where your embeddings and documents will be stored for retrieval.
            collection_name=COLLECTION_NAME,                  # Names the collection (or namespace) inside the Chroma vector store. Each collection is like a table or bucket that can be queried separately.
            embedding_function=ef,                            # Assigns the embedding model that will convert documents and queries into vector representations. In this case, ef is a Hugging Face embedding function.
            persist_directory=str(VECTORSTORE_DIR)            # Specifies the path where Chroma will save the vector database to disk. This makes the vector store persistent across sessions (it will be saved locally).
        )


def process_urls(urls, api_key):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """
    yield "Initializing Components"
    initialize_components(api_key)

    yield "Resetting vector store...✅"                      # Clears the existing Chroma collection.
    vector_store.reset_collection()

    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()                                     # Downloads and parses the web page content.

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE                                # Divides the data into smaller chunks for vector storage.
    )
    docs = text_splitter.split_documents(data)

    yield "Add chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)            # Stores chunks in Chroma with unique IDs.

    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources


if __name__ == "__main__":                                   # This line is a Python standard practice used to control the execution flow of your script.  It checks whether the script is being run directly or imported as a module in another script.
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for step in process_urls(urls, api_key):
        print(api_key)

    # process_urls(urls)
    answer, sources = generate_answer("Tell me what was the 30 year fixed mortagate rate along with the date?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
