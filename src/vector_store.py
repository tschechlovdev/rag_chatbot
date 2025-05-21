import logging

from pathlib import Path
from typing import List, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
import faiss
import bs4
from langchain import hub

def load_documents(data_path="data") -> List[Document]:
    documents = []
    for pdf_path in Path(data_path).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()[0:2] # TODO: shortcut for time reasons
        print(len(docs))
        documents.extend(docs)
    return documents

def split_documents(documents: List[Document], chunk_size=1000, chunk_overlap=500) -> List[Document]:
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(documents)

def create_vectorstore(self, docs: List[Document]) -> FAISS:
    return FAISS.from_documents(docs, self.embeddings)

def persist_vectorstore(self, db: FAISS) -> None:
    db.save_local(self.index_path)

def load_vectorstore(self) -> FAISS:
    return FAISS.load_local(self.index_path, self.embeddings)

model = "granite3.3"
llm = ChatOllama(model=model)
embeddings_model = OllamaEmbeddings(model=model)
VECTOR_STORE_PATH = Path("faiss_index")

def embed_documents(documents: List[Document]) -> List:
    return [embeddings_model.embed_query(doc.page_content) for doc in documents]

def save_embeddings_to_vector_db():
    FAISS.save_local(VECTOR_STORE_PATH)

def create_vector_store():
    print(VECTOR_STORE_PATH)
    if VECTOR_STORE_PATH.exists():
        return FAISS.load_local(str(VECTOR_STORE_PATH), embeddings=embeddings_model, allow_dangerous_deserialization=True)
    print(f"No Vectorstore found at {str(VECTOR_STORE_PATH)}")
    # Offline: Create a Knowledge Base
    documents = load_documents(data_path="data")
    splitted_documents = split_documents(documents=documents)
    #print(splitted_documents)
    doc_embeddings = embed_documents(splitted_documents)
    #print(doc_embeddings)

    embedding_dim = len(embeddings_model.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    vector_store.add_documents(splitted_documents)
    return vector_store

class VectorStore:

    def __init__(self, vector_store_path=VECTOR_STORE_PATH, llm_model="granite3.3",
                  chunk_size=500, chunk_overlap=50, persist=True, index_path="faiss_index"):
        self.vector_store_path = vector_store_path
        self.llm_model = llm_model
        self.embeddings_model = OllamaEmbeddings(model=llm_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist = persist
        self.index_path = index_path

        self._setup_vector_store()

    def _setup_vector_store(self):
        print(self.vector_store_path)
        if self.vector_store_path.exists():
            self.vector_store =  FAISS.load_local(str(self.vector_store_path), 
                                    embeddings=self.embeddings_model,
                                     allow_dangerous_deserialization=True)
        else:
            print(f"No Vectorstore found at {str(self.vector_store_path)}")

            self.embedding_dim = len(self.embeddings_model.embed_query("hello world"))
            self.index = faiss.IndexFlatL2(self.embedding_dim)

            self.vector_store = FAISS(
                embedding_function=embeddings_model,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            if self.persist:
                self.vector_store.save_local(self.index_path)

    def load_documents(self, data_path) -> List[Document]:
        documents = []
        for pdf_path in Path(data_path).glob("*.pdf"):
            docs = self.load_document(pdf_path)
            print(len(docs))
            documents.extend(docs)
        return documents

    def embed_documents(self, documents: List[Document]) -> List:
        return [embeddings_model.embed_query(doc.page_content) for doc in documents]
    
    def add_documents(self, documents: List[Document]) -> List:
        splitted_docs = self.chunk_documents(documents=documents)
        #doc_embeddings = self.embed_documents(splitted_docs)
        #print(len(doc_embeddings))
        #print(type(doc_embeddings))
        #self.vector_store.add_embeddings(doc_embeddings)
        self.vector_store.add_documents(splitted_docs)
        self.vector_store.save_local(self.index_path)
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                              chunk_overlap=self.chunk_overlap)    
        return text_splitter.split_documents(documents)
    
    def add_all_documents(self, data_path: str = "data"):
        documents = self.load_documents(data_path)
        return self.add_documents(documents)
    
    def load_document(self, pdf_path: Path) -> List[Document]:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()[0:2] # TODO: shortcut for time reasons
        return docs if isinstance(docs, list) else [docs]
    
    def add_document(self, filePath: Path):
        docs = self.load_document(filePath)
        return self.add_documents(docs)
    
    def similarity_search(self, question: str, k:int) -> List[Document]:
        return self.vector_store.similarity_search(question, k=k)
    
    def as_retriever(self):
        return self.vector_store.as_retriever()

    def index_websites(self, urls: list[str]):
        docs = self.website_to_documents(urls)
        return self.add_documents(docs)

    def website_to_documents(self, urls: list[str]) -> list[Document]:
        """Converts URLs to LangChain Document objects with metadata."""
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        
        print(docs)
        return docs