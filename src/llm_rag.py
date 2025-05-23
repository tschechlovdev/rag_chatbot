import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List
from langchain import hub
from langchain.schema import Document
from vector_store import VectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


class LLMRAGHandler:
    def __init__(self, model="granite3.3"):
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

        self.llm_chain = self.rag_prompt | self.llm | StrOutputParser()
        self.rag_chain = create_retrieval_chain(self.vector_store.as_retriever(), self.llm_chain)

    
    def generate_response(self, human_message) -> AIMessage:
        """Generates and appends a response from the LLM."""
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
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

    def get_history(self) -> List[BaseMessage]:
        return self.history
    
    def retrieve(self, question: str, k:int = 4) -> List[Document]:
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        return retrieved_docs

    
    def add_pdf_to_context(self, filePath: Path):
        self.vector_store.add_document(filePath)
    
if __name__ == '__main__':
    print(ChatOllama)