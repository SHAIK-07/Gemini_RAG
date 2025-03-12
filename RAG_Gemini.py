import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Streamlit UI setup
st.set_page_config(page_title="📄 Chat with PDF", layout="wide")
st.header("💬 Chat with Your PDF using Gemini AI")

# Sidebar: User API Key & PDF Upload
with st.sidebar:
    st.title("🔑 Enter API Key & Upload PDFs")

    # API Key Input
    api_key = st.text_input("Enter your Gemini API Key", type="password")

    if api_key:
        genai.configure(api_key=api_key)  # ✅ Ensure API Key is set
        st.success("✅ API Key Set Successfully!")

    # PDF Upload
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

    if st.button("Submit & Process") and api_key:
        with st.spinner("Processing PDFs..."):
            text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""  # ✅ Fix extraction issue

            text_chunks = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(text)

            # ✅ Pass API Key explicitly
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            st.success("✅ PDFs processed successfully!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_question = st.chat_input("Ask a question from the PDF...")

# Process user query
if user_question and api_key:
    with st.chat_message("user"):
        st.markdown(user_question)

    # ✅ Fix FAISS Deserialization Error
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    prompt_template = """
    Answer the question based on the provided context. If the answer is not in the context, say:
    "Answer is not available in the context."\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3, google_api_key=api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"]))

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_text = response["output_text"]

    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Store messages in session state
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": response_text})
