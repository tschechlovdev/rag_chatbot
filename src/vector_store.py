import logging

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
import faiss
import bs4

# Constants
model = "granite3.3"
llm = ChatOllama(model=model)
embeddings_model = OllamaEmbeddings(model=model)
VECTOR_STORE_PATH = Path("faiss_index")


class VectorStore:
    """
    A class for managing a vector store of documents.

    Attributes:
        vector_store_path (str): The path to the vector store.
        llm_model (str): The language model used for embeddings.
        embeddings_model (OllamaEmbeddings): The embeddings model.
        chunk_size (int): The size of chunks for document splitting.
        chunk_overlap (int): The overlap between chunks for document splitting.
        persist (bool): Whether to persist the vector store to disk.
        index_path (str): The path to save the index.

    Methods:
        __init__(vector_store_path=VECTOR_STORE_PATH, llm_model="granite3.3",
                  chunk_size=500, chunk_overlap=50, persist=True, index_path="faiss_index"):
            Initializes the VectorStore object.

        _setup_vector_store():
            Sets up the vector store, either loading from disk or creating a new one.

        load_documents(data_path) -> List[Document]:
            Loads documents from the specified directory.

        add_documents(documents: List[Document]) -> List[Document]:
            Adds documents to the vector store and saves the index.

        chunk_documents(documents: List[Document]) -> List[Document]:
            Splits documents into smaller chunks.

        add_all_documents(data_path: str = "data") -> List[Document]:
            Loads and adds all documents from the specified directory.

        load_document(pdf_path: Path) -> List[Document]:
            Loads a single document from a PDF file.

        add_document(filePath: Path) -> List[Document]:
            Adds a single document from a PDF file.

        similarity_search(question: str, k:int) -> List[Document]:
            Performs a similarity search based on the given question.

        as_retriever() -> VectorStoreRetriever:
            Returns a retriever object for the vector store.

        index_websites(urls: list[str]) -> List[Document]:
            Indexes documents from the given URLs.

        website_to_documents(urls: list[str]) -> list[Document]:
            Converts URLs to LangChain Document objects with metadata.
    """
    def __init__(self, vector_store_path=VECTOR_STORE_PATH, llm_model="granite3.3",
                  chunk_size=500, chunk_overlap=50, persist=True, index_path="faiss_index"):
        """
        Initialize the VectorStore class.

        Parameters:
        vector_store_path (str): The path to the directory where the vector store will be saved.
        llm_model (str): The name of the language model to use for generating embeddings.
        chunk_size (int): The number of documents to process at a time.
        chunk_overlap (int): The number of documents to overlap between chunks.
        persist (bool): Whether to persist the vector store to disk.
        index_path (str): The path to the Faiss index file.

        Attributes:
        vector_store_path (str): The path to the directory where the vector store will be saved.
        llm_model (str): The name of the language model to use for generating embeddings.
        embeddings_model (OllamaEmbeddings): The embeddings model initialized with the specified language model.
        chunk_size (int): The number of documents to process at a time.
        chunk_overlap (int): The number of documents to overlap between chunks.
        persist (bool): Whether to persist the vector store to disk.
        index_path (str): The path to the Faiss index file.
        """
        self.vector_store_path = vector_store_path
        self.llm_model = llm_model
        self.embeddings_model = OllamaEmbeddings(model=llm_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist = persist
        self.index_path = index_path

        self._setup_vector_store()

    def _setup_vector_store(self) -> None:
        """
        Sets up the vector store for the model.

        This method checks if a vector store already exists at the specified path.
        If it does, it loads the existing vector store using the provided embeddings model.
        If not, it creates a new vector store using the provided embeddings model and
        persists it to the specified path if the 'persist' attribute is set to True.

        Args:
            self: An instance of the class containing the following attributes:
                - vector_store_path (Path): The path where the vector store is stored.
                - embeddings_model (EmbeddingsModel): The model used for generating embeddings.
                - persist (bool): Whether to persist the vector store to disk.

        Returns:
            None
        """
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
        """
        Loads documents from the specified directory.

        Args:
            data_path (str): The path to the directory containing PDF files.

        Returns:
            List[str]: A list of document texts, each representing a document loaded from a PDF file.
        """
        documents = []
        for pdf_path in Path(data_path).glob("*.pdf"):
            docs = self.load_document(pdf_path)
            print(len(docs))
            documents.extend(docs)
        return documents
    
    def add_documents(self, documents: List[Document]) -> List[Document]:
        """
        Adds a list of documents to the vector store after splitting them into chunks.

        Args:
            documents (List[Document]): The list of documents to be added.

        Returns:
            List[Document]: The list of document chunks that were added to the vector store.
        """
        splitted_docs = self.chunk_documents(documents=documents)
        self.vector_store.add_documents(splitted_docs)
        self.vector_store.save_local(self.index_path)
        return splitted_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunks using a recursive character text splitter.

        Args:
            documents (List[Document]): A list of Document objects to be chunked.

        Returns:
            List[Document]: A list of chunked Document objects.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                              chunk_overlap=self.chunk_overlap)    
        return text_splitter.split_documents(documents)
    
    
    def load_document(self, pdf_path: Path) -> List[Document]:
        """
        Loads and returns the content of a PDF document as a list of Document objects.

        Args:
            pdf_path (Path): The file path to the PDF document.

        Returns:
            List[Document]: A list containing up to the first two Document objects extracted from the PDF.
        """
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        return docs if isinstance(docs, list) else [docs]
    
    def add_document(self, filePath: Path) -> List[Document]:
        """
        Adds a document to the vector store from the specified file path.

        Args:
            filePath (Path): The path to the file to be loaded and added.

        Returns:
            List[Document]: A list of Document objects that were added to the vector store.
        """
        docs = self.load_document(filePath)
        return self.add_documents(docs)
    
    def similarity_search(self, question: str, k:int) -> List[Document]:
        """
        Performs a similarity search on the vector store using the provided question.

        Args:
            question (str): The input query string to search for similar documents.
            k (int): The number of top similar documents to retrieve.

        Returns:
            List[Document]: A list of the top-k documents most similar to the input question.
        """
        return self.vector_store.similarity_search(question, k=k)
    
    def as_retriever(self) -> VectorStoreRetriever:
        """
        Returns a VectorStoreRetriever instance for retrieving documents from the underlying vector store.

        :returns: A VectorStoreRetriever object that can be used to perform retrieval operations.
        :rtype: VectorStoreRetriever
        """
        return self.vector_store.as_retriever()

    def index_websites(self, urls: list[str]) -> List[Document]:
        """
        This method converts the URLs to Document objects, adds them to the vector store,
        and returns the list of Document objects created from the indexed websites.

        Args:
            urls (list[str]): A list of website URLs to be indexed.

        Returns:
            List[Document]: A list of Document objects created from the indexed websites.
        """
        docs = self.website_to_documents(urls)
        return self.add_documents(docs)

    def website_to_documents(self, urls: list[str]) -> list[Document]:
        """
        Loads and parses web pages from the given list of URLs into Document objects.
        Args:
            urls (list[str]): A list of website URLs to load and parse.
        Returns:
            list[Document]: A list of Document objects extracted from the specified web pages.
        Notes:
            Only HTML elements with the classes "post-content", "post-title", or "post-header"
            are parsed from each web page.
        """
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        return docs