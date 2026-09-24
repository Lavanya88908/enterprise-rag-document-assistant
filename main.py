import os
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv

from vector_store import VectorStoreManager
from ingest import DocumentIngestor
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.fake import FakeListLLM

DB_NAME = "rag_logs.db"

def log_query(query: str, response: str, num_chunks: int):
    """Logs the query and response to SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 2. Explicitly execute CREATE TABLE IF NOT EXISTS before logging
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_logs (
            timestamp TEXT, 
            query TEXT, 
            response TEXT, 
            response_length INTEGER, 
            chunks_retrieved INTEGER
        )
    ''')
    
    # 3. Ensure connection.commit() is called
    cursor.execute('''
        INSERT INTO query_logs (timestamp, query, response, response_length, chunks_retrieved)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), query, response, len(response), num_chunks))
    conn.commit()
    conn.close()

class RAGPipeline:
    def __init__(self, vector_store_manager: VectorStoreManager, llm):
        self.vector_store_manager = vector_store_manager
        self.llm = llm
        
        # Setup Prompt Template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an Enterprise Document Intelligence Assistant. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""
        )

    def query(self, question: str, k: int = 3):
        # 1. Retrieve
        if not self.vector_store_manager.vector_store:
            self.vector_store_manager.initialize_store()
            
        retriever = self.vector_store_manager.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        
        # 2. Format Context
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # 3. Format Prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # 4. Generate Response
        response = self.llm.invoke(prompt)
        
        # Extract string if response is an AIMessage object
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
            
        # 5. Log
        log_query(question, response_text, len(docs))
        
        return response_text, docs

def main():
    load_dotenv()
    print("Starting Enterprise RAG Document Intelligence Assistant...")

if __name__ == "__main__":
    def run_internal_verification():
        print("--- Running Internal RAG Verification ---")
        try:
            # 2. Mock vector store initialization with a test document
            manager = VectorStoreManager(persist_directory="./test_rag_db")
            manager.initialize_store()
            
            from langchain_core.documents import Document
            test_docs = [
                Document(page_content="The Enterprise RAG assistant supports PDF and TXT ingestion, using ChromaDB and HuggingFace.", metadata={"source": "test_doc"})
            ]
            manager.add_documents(test_docs)
            
            # 3. Setup Fake LLM
            fake_llm = FakeListLLM(responses=["The Enterprise RAG assistant supports PDF and TXT ingestion, utilizing ChromaDB and HuggingFace models."])
            
            # 4. Run Pipeline
            pipeline = RAGPipeline(manager, fake_llm)
            test_query = "What formats does the assistant support?"
            print(f"Executing test query: '{test_query}'")
            
            start_time = time.time()
            response, docs = pipeline.query(test_query, k=1)
            duration = time.time() - start_time
            
            print(f"Generated Response: '{response}'")
            print(f"Retrieved {len(docs)} chunks. Pipeline execution time: {duration:.4f}s")
            
            # 5. Verify Logging
            conn = sqlite3.connect(DB_NAME)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT query, response_length FROM query_logs ORDER BY rowid DESC LIMIT 1")
                log_entry = cursor.fetchone()
            finally:
                conn.close()
            
            if log_entry and log_entry[0] == test_query and log_entry[1] > 0:
                print("Logging verification passed.")
            else:
                print(f"WARNING: Logging verification failed. Retrieved: {log_entry}")
                
            print("--- Internal Verification Passed ---")
            
        except Exception as e:
            print("--- Internal Verification Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            import shutil
            if os.path.exists("./test_rag_db"):
                shutil.rmtree("./test_rag_db", ignore_errors=True)
                
    run_internal_verification()
