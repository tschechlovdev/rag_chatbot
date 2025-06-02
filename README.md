# RAG-Chatbot — Chat with Your Own Documents


This project lets you run a fully local **RAG-based (Retrieval-Augmented Generation)** chatbot using your own PDFs or web content. Ask questions in natural language, and get answers based on the actual contents of your documents.

![1_tEelGLOyg6n7oUJ0a1fMUA](https://github.com/user-attachments/assets/ec00a9d7-53f7-4c52-b51c-0c565f92521c)

It uses the following tools for this:
- [LangChain](https://www.langchain.com/) for orchestration
- [FAISS](https://github.com/facebookresearch/faiss) for semantic vector search
- [Ollama](https://ollama.com) to run open-source LLMs locally
- [Streamlit](https://streamlit.io) for an easy-to-use chat interface

## ✨ Features

- 📄 Upload PDFs or URLs as your data source
- 🧠 Store document chunks as embeddings in a FAISS vector store
- 🔍 Retrieve relevant content using semantic similarity search
- 💬 Generate context-aware answers via local LLM
- 💻 All running 100% locally

---

## 🚀 Getting Started

### 1. Install Ollama and Run a Local LLM

Make sure you have [Ollama](https://ollama.com) installed and a compatible model (e.g. `granite:3.3`) downloaded.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

# Download model
```bash
ollama pull granite:3.3
```

# Start the model
```bash 
ollama run granite:3.3
```

### 2. Clone This Repository
```bash
git clone https://github.com/yourname/rag-chatbot.git
cd rag_chatbot
```

### 3. Install Python Dependencies

This project uses Python 3.9+.

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```streamlit run app.py```

## 📂 File Upload
Once the app is running: Go to the sidebar to upload one or more PDF files.

Ask natural questions about the content in the chat interface.

The chatbot will search for relevant sections and answer using context.

## 🏛️ Architecture Overview
![1_gXq3HJeXbPO2aGgFDYh0TA](https://github.com/user-attachments/assets/b492d7a7-d280-40ff-b92b-534cd1c415e7)

- **ChatUI**: The user interface is built with Streamlit, using its built-in chat_message components to create a conversational layout. Users can upload documents in the sidebar and interact with the chatbot in real time.
- **LLMRAGHandler**: This is the main component that connects everything. It is implemented using LangChain and is responsible for managing the conversation flow, retrieving relevant context from the vector store, formatting prompts using a custom template, calling the LLM, and caching chat history.
- **Vector Store**: Responsible for storing the documents as vector embeddings in FAISS, a high-speed similarity search library and retrieving the relevant context
-  **LLM**: The chatbot runs the Granite 3.3 model locally using Ollama. This means: Easy setup and prototyping, easy model switching, and full control over your data (everything stays local
- **Conversation Store**: To make the chatbot stateful, we store the conversation history in a local file (e.g. JSON). This allows the chat to resume where you left off - even after refreshing the browser.
  
## ⚠️ Limitations
- Initial PDF parsing and embedding may take a few seconds for large files.
- Latency depends on the chosen LLM model.
- Evaluation of answers is qualitative — no scoring function included.
- Runs only locally for easier development

## 💡 Ideas for Future Improvements
- Use agentic RAG (history-aware retrievers, dynamic tool-calling)
- Tool Calling
- Other Data Sources (Google Drive, Notion, ...)
- Cloud deployment
- UI enhancements and document summarization


## 📄 License
MIT License. See LICENSE for details.
