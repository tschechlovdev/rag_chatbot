import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from llm_rag import LLMRAGHandler
from conversation import ConversationManager
from langchain_community.vectorstores import FAISS
from pathlib import Path

def process_new_pdfs(uploaded_files):
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    for file in uploaded_files:
        print(st.session_state.processed_files)
        print(file.name)
        if file.name in st.session_state.processed_files:
            continue

        save_path = UPLOAD_DIR / file.name
        with save_path.open("wb") as f:
            f.write(file.read())

        st.sidebar.success(f"{file.name} saved.")

        with st.spinner(f"{file.name} is processed..."):
            st.session_state.llm.add_pdf_to_context(save_path)

        st.session_state.processed_files.add(file.name)
        st.sidebar.success(f"{file.name} was indexed.")
        st.rerun()

conversation_manager = ConversationManager()
st.set_page_config(page_title="RAG Chatbot")
st.title("🤖 Chat with your PDFs 📄")

# Folder for saving uploaded PDFs
UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

#########################
######### Initialization

# Initialize session state
if "llm" not in st.session_state:
    st.session_state.llm = LLMRAGHandler()
    saved_conversation = conversation_manager.load()
    if saved_conversation:
        st.session_state.llm.history = saved_conversation

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {p.name for p in UPLOAD_DIR.glob("*.pdf")}


st.sidebar.subheader("📁 Already Saved PDFs:")
if st.session_state.processed_files:
    for pdf_path in st.session_state.processed_files :
        st.sidebar.markdown(f"- {pdf_path}")
else:
    st.sidebar.info("No PDFs uploaded yet.")
#########################




#########################
###### PDF Upload
st.sidebar.header("📄 Upload PDFs")

# Hochladen der PDF-Dateien
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# Speichern & Anzeigen der Dateinamen
# TODO: These files should be directly loaded from vector store!
# TODO: Hier werden alle hochgeladenen files verarbeitet

if uploaded_files:
    process_new_pdfs(uploaded_files)

st.sidebar.header("🌐 Add Website URLs")
urls = st.sidebar.text_area("Enter website URLs (one per line)").splitlines()
if st.sidebar.button("📥 Add websites"):
    with st.spinner("Processing websites..."):
        print(urls)
        st.session_state.llm.vector_store.index_websites(urls)
    st.sidebar.success("Websites indexed successfully.")
#########################


#####################
##### Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    #response = st.session_state.llm.ask(user_input)
    
    with st.spinner("Thinking..."):
        response = st.session_state.llm.generate_response(user_input)
        print(f"Response: {response}")
        print(type(response))
conversation_manager.save(st.session_state.llm.get_history())

# Display chat messages
for msg in st.session_state.llm.get_history():
    if isinstance(msg, SystemMessage):
        # do not display instructions for the chatbot
        continue
    
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
#########################


#########################
# Sidebar actions
if st.sidebar.button("🗑️ Reset Conversation"):
    st.session_state.llm.reset()
    conversation_manager.clear()
    st.rerun()