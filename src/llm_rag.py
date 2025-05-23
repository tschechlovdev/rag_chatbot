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
        
        # New Version:
        #qa_prompt = ChatPromptTemplate.from_messages([
        #        ("system", self.system_prompt),
        #    MessagesPlaceholder("chat_history"),
        #    ("ai", "{context}"),#

        # ("human", "{input}")])
        
        #self.history_aware_retriever = create_history_aware_retriever(self.llm, 
        #                                                              self.vector_store.as_retriever(),
        #                                                                qa_prompt)
        
        #self.qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        # Only create rag_chain if dependencies are defined
        #if self.history_aware_retriever and self.qa_chain:
        self.llm_chain = self.rag_prompt | self.llm | StrOutputParser()
        self.rag_chain = create_retrieval_chain(self.vector_store.as_retriever(), self.llm_chain)

        #self.store={}
        #else:
         #   self.rag_chain = None


    def get_chat_history(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
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

        #context_docs = self.retrieve(human_message)
        #response = self.generate(question=human_message, context=context_docs)

        #if isinstance(human_message, str):
        #    human_message = HumanMessage(content=human_message)
        self.history.append(HumanMessage(content=human_message))                
        self.history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    def reset(self) -> None:
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

    def get_history(self) -> List[str]:
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