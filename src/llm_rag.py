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


# prompt template for q&a with rag


class LLMRAGHandler:
    def __init__(self, model="granite3.3"):
        self.llm = ChatOllama(model=model)
        self.vector_store = VectorStore(llm_model=model)
        self.system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, try to answer the question without context but mention that the context does not provide enough information." \
        " Use three sentences maximum and keep the answer concise."
        self.history = [SystemMessage(content=self.system_prompt)]
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
        # TODO: Ggfs. retrieval chain verwenden?
        context_docs = self.retrieve(human_message)
        response = self.generate(question=human_message, context=context_docs)
        self.history.append(human_message)                
        self.history.append(response)
        return response.content

    def reset(self) -> None:
        self.history = []

    def get_history(self) -> List[BaseMessage]:
        return self.history
    
    def retrieve(self, question: str):
        retrieved_docs = self.vector_store.similarity_search(question)
        return retrieved_docs

    def generate(self, question: str, context: List[Document]):
        docs_content = "\n\n".join(doc.page_content for doc in context)
        messages = self.rag_prompt.invoke({"question": question, "context": docs_content, "chat_history": self.history}).to_messages()
        response = self.llm.invoke(messages)
        print(f"Response: {response}")
        print(f"Content of Response: {response.content}")
        return response
    
    def add_pdf_to_context(self, filePath: Path):
        self.vector_store.add_document(filePath)
    
if __name__ == '__main__':
    print(ChatOllama)