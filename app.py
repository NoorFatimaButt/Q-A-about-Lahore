# Import required libraries
import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Client

# Function to query the Groq API using the Groq library
def query_groq_api(prompt):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,  # Use your model here, e.g., "gemma-7b-it"
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Function to load LLM
def load_llm():
    def groq_pipeline(prompt):
        response = query_groq_api(prompt)
        return response
    return groq_pipeline

# Streamlit app starts here
st.title("Lahore History Search App")
st.write("Explore the history of Lahore using advanced LLM capabilities and retrieved information from web and PDF sources.")

# Input field for user query
user_query = st.text_input("Enter your query about Lahore's history:", "")

# Set up constants and configurations
API_KEY = "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"
MODEL_NAME = "gemma-7b-it"
client = Client(api_key=API_KEY)
groq_llm = load_llm()

# Wikipedia Tool Initialization
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Load web documents
web_loader = WebBaseLoader("https://giu.edu.pk/history-lahore")
web_docs = web_loader.load()

# Specify the PDF file path in the 'data' directory
pdf_file_path = "data/lahore.pdf"

# Load PDF documents
pdf_documents = []
try:
    with st.spinner("Processing PDF..."):
        pdf_loader = PyPDFLoader(pdf_file_path)
        pdf_docs = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pdf_documents = text_splitter.split_documents(pdf_docs)
except FileNotFoundError:
    st.warning(f"PDF file not found at: {pdf_file_path}. Please check the file path.")

# Split web documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
web_documents = text_splitter.split_documents(web_docs)

# Combine all documents
all_documents = web_documents + pdf_documents

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Create FAISS vector store
vectordb = FAISS.from_documents(all_documents, embedding=embedding_model)

# Create a unified retriever
retriever = vectordb.as_retriever()

# Retrieve relevant documents and generate response
def groq_qa_chain(prompt):
    documents = retriever.get_relevant_documents(prompt)
    context = "\n".join([doc.page_content for doc in documents])
    query = f"Context: {context}\n\nQuestion: {prompt}"
    response = groq_llm(query)
    return response

# Query handling with button
if st.button("Get Answer"):
    if user_query:
        with st.spinner("Fetching response..."):
            answer = groq_qa_chain(user_query)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a query to get started.")
else:
    st.write("Press the button to get an answer.")
