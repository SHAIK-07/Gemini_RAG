# 📄 Chat with PDF using Gemini AI

This is a **Streamlit** application that allows users to upload PDFs and ask questions based on their content using **Google Gemini AI**. The application extracts text from PDFs, processes it into embeddings using **FAISS**, and enables interactive Q&A with **Gemini Pro**.

---

## 🚀 Features
- **Upload Documents (PDF, CSV, Excel) or Enter a URL**: Extracts text from uploaded files.
- **Text Chunking**: Splits extracted text into manageable chunks for efficient processing.
- **Embeddings with FAISS**: Uses Google's `embedding-001` model to store and retrieve text chunks.
- **Conversational AI**: Uses `gemini-1.5-pro-latest` to generate responses based on document content.
- **Chat History**: Maintains chat history for a smooth user experience.
- **API Key Input**: Users can enter their own **Google Gemini API key** for authentication.

---

## 🛠️ Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/chat-with-pdf-gemini.git
cd chat-with-pdf-gemini
```

### **2️⃣ Create a Virtual Environment & Install Dependencies**
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Activate the virtual environment (Mac/Linux)
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

### **3️⃣ Set Up Google API Key**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to generate an API key.
2. Ensure the **Gemini API** is enabled for the key.
3. Copy your API key and enter it in the Streamlit app sidebar.

---

## ▶️ Running the Application
```bash
streamlit run app.py
```
This will launch the application in your web browser.

---

## 📌 Usage Guide
1. **Enter your Google Gemini API Key** in the sidebar.
2. **Upload Documents (PDF, CSV, Excel) or Enter a URL**.
3. Click on **Submit & Process** to extract and index text.
4. Type a **question related to the PDF content** in the chat input.
5. Get AI-generated answers based on document content.

---

## 🔧 Troubleshooting
### **Invalid API Key Error**
- Ensure you’re using the correct API key.
- Enable **Gemini API** in Google AI Studio.
- Try generating a **new API key** if issues persist.

### **Text Extraction Issues**
- Some Files may have non-selectable text (scanned documents). Use **OCR tools** like Tesseract for preprocessing.

---

## 📜 License
This project is open-source under the **MIT License**.

---

## 👨‍💻 Author
Developed by [Shaik Hidaythulla](https://www.linkedin.com/in/shaik-hidaythulla/) Feel free to reach out for any queries or improvements!

Happy coding! 🚀

