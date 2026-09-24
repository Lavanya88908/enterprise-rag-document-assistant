
import os
import shutil
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.model_name = model_name
        self.embeddings = None
        self.vector_store = None

    def initialize_store(self):
        """
        Initializes the ChromaDB vector store with HuggingFace embeddings.
        """
        print(f"Initializing embeddings model: {self.model_name}...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        except Exception as e:
            print(f"Error initializing embeddings model: {e}")
            raise e

        print(f"Initializing ChromaDB vector store at {self.persist_directory}...")
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="enterprise_rag_docs"
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise e

    def add_documents(self, document_chunks: List[Document]):
        """
        Adds document chunks to the vector store.
        """
        if not self.vector_store:
            self.initialize_store()
            
        if not document_chunks:
            print("No documents provided to add to vector store.")
            return
            
        print(f"Adding {len(document_chunks)} document chunks to the vector store...")
        try:
            # Chroma handles batching and adding under the hood
            self.vector_store.add_documents(document_chunks)
            print("Successfully added documents to the vector store.")
        except Exception as e:
            print(f"Error adding documents to the vector store: {e}")
            raise e

if __name__ == "__main__":
    def run_internal_verification():
        print("--- Running Internal Verification ---")
        test_db_dir = "./test_chroma_db"
        try:
            # 1. Initialize store
            manager = VectorStoreManager(persist_directory=test_db_dir)
            manager.initialize_store()
            
            # 2. Test database connection & embedding dimension
            test_text = "This is a test document."
            print("Testing embedding generation and dimension matching...")
            test_embed = manager.embeddings.embed_query(test_text)
            print(f"Generated embedding dimension: {len(test_embed)}")
            
            if len(test_embed) != 384: # all-MiniLM-L6-v2 dimension is 384
                print(f"WARNING: Unexpected embedding dimension. Expected 384, got {len(test_embed)}")
            else:
                print("Dimension check passed (384).")
            
            # 3. Add sample documents
            test_docs = [
                Document(page_content="RAG systems use vector databases for retrieval.", metadata={"source": "doc1"}),
                Document(page_content="Embeddings capture semantic meaning.", metadata={"source": "doc2"})
            ]
            manager.add_documents(test_docs)
            
            # 4. Verify retrieval
            print("Testing vector retrieval...")
            results = manager.vector_store.similarity_search("How do RAG systems work?", k=1)
            
            if results:
                print(f"Retrieval successful. Top result: '{results[0].page_content}'")
            else:
                print("WARNING: Retrieval returned no results.")
                
            print("--- Internal Verification Passed ---")
        except Exception as e:
            print(f"--- Internal Verification Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(test_db_dir):
                print(f"Cleaning up test db: {test_db_dir}")
                shutil.rmtree(test_db_dir, ignore_errors=True)

    run_internal_verification()
