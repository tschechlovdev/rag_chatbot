from pathlib import Path
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List
from langchain.schema import Document
from vector_store import VectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

class LLMRAGHandler:
    """
    A class to handle LLM-based RAG (Retrieval-Augmented Generation) tasks.
    
    Attributes:
        llm (ChatOllama): The language model used for generating responses.
        vector_store (VectorStore): The vector store used for document retrieval.
        system_prompt (str): The system prompt given to the model.
        history (List[BaseMessage]): The conversation history.
        rag_prompt (PromptTemplate): The prompt template for q&a with RAG.
        llm_chain (Chain): The chain for RAG.
        rag_chain (Chain): The retrieval chain.
    
    Methods:
        __init__(self, model="granite3.3"): Initializes the LLMRAGHandler with the specified model.
        generate_response(self, human_message) -> AIMessage: Generates and appends a response from the LLM.
        reset(self) -> None: Resets the conversation history.
        get_history(self) -> List[BaseMessage]: Returns the conversation history.
        retrieve(self, question: str, k:int = 4) -> List[Document]: Retrieves the most relevant documents for a given question.
        add_pdf_to_context(self, filePath: Path): Adds a PDF file to the context for retrieval.

    """
    def __init__(self, model="granite3.3"):
        """        
        Initializes the LLMRAGHandler with the specified model.

        Args:
            model (str): The model to use for the language model and vector store. Default is "granite3.3".
        """
        self.llm = ChatOllama(model=model)
        self.vector_store = VectorStore(llm_model=model)
        
        # System prompt - These are the instructions for the model
        self.system_prompt = "You are an assistant for question-answering tasks." \
        " Use the following pieces of retrieved context to answer the question. " \
        "If you don't know the answer, try to answer the question without context but mention that " \
        "the context does not provide enough information." \
        " Use three sentences maximum and keep the answer concise."
        
        # keep track of the conversation history
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

        # prompt template for q&a with rag
        self.rag_prompt = PromptTemplate.from_template(
        "Previous conversation: {chat_history}"
        " Question: {input}" \
        " Context: {context}" \
        " Answer:")

        # Chain for querying the LLM and getting the answer
        self.llm_chain = self.rag_prompt | self.llm | StrOutputParser()

        # Create retrieval chain for RAG 
        self.rag_chain = create_retrieval_chain(self.vector_store.as_retriever(), self.llm_chain)

    
    def generate_response(self, human_message) -> AIMessage:
        """
        Generates and appends a response from the LLM.

        Args:
            human_message (str): The user's message.

        Returns:
            AIMessage: The AI's response.
        """
        print(f"Adding Humang Message...")
        print(f"{human_message}")

        print("Generating response from LLM...")
        context_docs = self.retrieve(human_message)
        # Run the retrieval chain
        response = self.rag_chain.invoke({
            "input": human_message,
            "context": context_docs,
            "chat_history": self.history}
        )
        print(response)

        self.history.append(HumanMessage(content=human_message))                
        self.history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    def reset(self) -> None:
        """
        Resets the conversation history.
        """
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

    def get_history(self) -> List[BaseMessage]:
        """
        Returns the conversation history.

        Returns:
            List[BaseMessage]: The conversation history.
        """       
        return self.history
    
    def retrieve(self, question: str, k:int = 4) -> List[Document]:
        """
        Retrieves the most relevant documents for a given question.

        Args:
            question (str): The question to retrieve documents for.
            k (int): The number of documents to retrieve. Default is 4.

        Returns:
            List[Document]: The retrieved documents.
        """
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        return retrieved_docs

    
    def add_pdf_to_context(self, filePath: Path) -> List[Document]:
        """
        Adds a PDF file to the context for retrieval.

        Args:
            filePath (Path): The path to the PDF file.
        Returns:
            List[Document]: The documents added to the vector store.
        """
        self.vector_store.add_document(filePath)
    
if __name__ == '__main__':
    print(ChatOllama.list_models())