import os
import re
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentIngestor:
    def __init__(self, source_directory: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.source_directory = source_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing extra whitespace and fixing common formatting issues.
        """
        if not text:
            return ""
        # Collapse multiple whitespaces and newlines into single spaces/newlines
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def load_documents(self) -> List[Document]:
        """
        Loads documents from the source directory. Supports TXT and PDF files.
        """
        if not os.path.exists(self.source_directory):
            print(f"Warning: Source directory '{self.source_directory}' does not exist.")
            return []

        print(f"Loading documents from {self.source_directory}...")
        documents = []
        
        # Load Text files safely
        try:
            txt_loader = DirectoryLoader(
                self.source_directory, 
                glob="**/*.txt", 
                loader_cls=TextLoader,
                loader_kwargs={'autodetect_encoding': True}
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
        except Exception as e:
            print(f"Error loading text files: {e}")

        # Load PDF files safely
        try:
            pdf_loader = DirectoryLoader(
                self.source_directory, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
        except Exception as e:
            print(f"Error loading PDF files: {e}")

        # Clean document content safely
        cleaned_documents = []
        for doc in documents:
            try:
                cleaned_content = self.clean_text(doc.page_content)
                if cleaned_content: # Ignore empty documents after cleaning
                    doc.page_content = cleaned_content
                    cleaned_documents.append(doc)
            except Exception as e:
                print(f"Error cleaning document {doc.metadata.get('source', 'unknown')}: {e}")
            
        print(f"Successfully loaded and cleaned {len(cleaned_documents)} documents.")
        return cleaned_documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits loaded documents into smaller chunks for embedding.
        """
        if not documents:
            print("No documents to chunk.")
            return []
            
        print(f"Chunking documents with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        try:
            chunks = text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
            return chunks
        except Exception as e:
            print(f"Error during document chunking: {e}")
            return []

if __name__ == "__main__":
    # Internal Verification Test
    print("--- Running Internal Verification ---")
    test_dir = "sample_data"
    os.makedirs(test_dir, exist_ok=True)
    test_file_path = os.path.join(test_dir, "test.txt")
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        # Create a document of about 1250 characters
        f.write("This is a test document. " * 50)
        
    ingestor = DocumentIngestor(source_directory=test_dir, chunk_size=500, chunk_overlap=50)
    docs = ingestor.load_documents()
    chunks = ingestor.chunk_documents(docs)
    
    if docs:
        print(f"Test Doc size: {len(docs[0].page_content)}")
    for i, c in enumerate(chunks):
        print(f"Chunk {i} size: {len(c.page_content)}")
        if len(c.page_content) > 500:
            print(f"WARNING: Chunk {i} exceeds max chunk size!")
    
    # Cleanup
    os.remove(test_file_path)
    os.rmdir(test_dir)
    print("--- Internal Verification Complete ---")
