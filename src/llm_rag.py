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
        self.history = [SystemMessage(content=self.system_prompt)]

        # prompt template for q&a with rag
        self.rag_prompt = PromptTemplate.from_template(
        "Previous conversation: {chat_history}"
        " Question: {question}" \
        " Context: {context}" \
        " Answer:")

    def generate_response(self, human_message) -> AIMessage:
        """Generates and appends a response from the LLM."""
        print(f"Adding Humang Message...")
        print(f"{human_message}")

        print("Generating response from LLM...")

        context_docs = self.retrieve(human_message)
        response = self.generate(question=human_message, context=context_docs)
        if isinstance(human_message, str):
            human_message = HumanMessage(content=human_message)
        self.history.append(human_message)                
        self.history.append(response)
        return response

    def reset(self) -> None:
        self.history = []

    def get_history(self) -> List[BaseMessage]:
        return self.history
    
    def retrieve(self, question: str, k:int = 4):
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        return retrieved_docs

    def generate(self, question: str, context: List[Document]):
        docs_content = "\n\n".join(doc.page_content for doc in context)
        messages = self.rag_prompt.invoke({"question": question, 
                                           "context": docs_content, 
                                           "chat_history": self.history}).to_messages()
        response = self.llm.invoke(messages)
        print(f"Type of response: {type(response)}")
        print(f"Response: {response}")
        print(f"Content of Response: {response.content}")
        return response
    
    def add_pdf_to_context(self, filePath: Path):
        self.vector_store.add_document(filePath)
    
if __name__ == '__main__':
    print(ChatOllama)