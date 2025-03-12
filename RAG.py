import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Streamlit UI setup
st.set_page_config(page_title="📝 Chat with Documents", layout="wide")
st.header("💬 Chat with Your Documents using Gemini AI")

# Sidebar: API Key & File Upload
with st.sidebar:
    st.title("🔑 Enter API Key & Upload Files")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ API Key Set Successfully!")
    
    source_option = st.selectbox("Select Source Type", ["PDF", "URL", "Excel/CSV"])
    
    text = ""
    
    if source_option == "PDF":
        uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        
        if st.button("Submit & Process") and api_key:
            with st.spinner("Processing PDFs..."):
                try:
                    for pdf in uploaded_files:
                        pdf_reader = PdfReader(pdf)
                        for page in pdf_reader.pages:
                            extracted_text = page.extract_text() or ""
                            text += extracted_text.encode("utf-8", "ignore").decode("utf-8")
                    
                except Exception as e:
                    st.error("❌ Error processing PDFs. Please check the file format.")

    elif source_option == "URL":
        url = st.text_input("Enter a URL")
        if st.button("Fetch & Process") and api_key and url:
            with st.spinner("Fetching URL..."):
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text()
                    
                except Exception as e:
                    st.error("❌ Error fetching the URL. Please check the link.")
    
    elif source_option == "Excel/CSV":
        uploaded_file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])
        
        if st.button("Submit & Process") and api_key and uploaded_file:
            with st.spinner("Processing File..."):
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    text = df.to_string()
                except Exception as e:
                    st.error("❌ Error processing the file. Ensure it is a valid CSV or Excel file.")
    
    if text:
        text_chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("✅ Data processed successfully!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask a question from the document...")

if user_question and api_key:
    with st.chat_message("user"):
        st.markdown(user_question)
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        prompt_template = """
        Answer the question based on the provided context. If the answer is not in the context, say:
        "Answer is not available in the context."
        
        Context:\n {context}\n
        Question:\n {question}\n
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=api_key)
        chain = load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"]))
        
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        response_text = response.get("output_text", "I'm sorry, I couldn't generate a response.")
    
    except Exception as e:
        response_text = "An error occurred while processing your question. Please try again."
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": response_text})

st.sidebar.markdown("Powered by [GenAI](https://genai.ai/)")