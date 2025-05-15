import os
import tempfile
from typing import List, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(uploaded_files: List[tempfile.SpooledTemporaryFile]) -> List[Document]:
    """
    Loads documents from a list of uploaded files.
    Supports PDF and TXT files.

    Args:
        uploaded_files: A list of file-like objects (from Streamlit uploader).

    Returns:
        A list of LangChain Document objects.
    """
    all_documents = []
    for source_file in uploaded_files:
        # Get the original filename
        source_filename = source_file.name
        
        # Create a temporary file to store the uploaded content because many LangChain loaders expect a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source_filename)[1]) as tmp_file:
            tmp_file.write(source_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if source_filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_file_path)
                # PyPDFLoader loads each page as a separate document
                # It automatically adds 'source' and 'page' to metadata
                docs_from_file = loader.load() 
                # We can update the 'source' metadata to be more specific if needed
                for doc in docs_from_file:
                    doc.metadata["source"] = source_filename # Ensure source is the original filename
                all_documents.extend(docs_from_file)
            elif source_filename.lower().endswith(".txt"):
                loader = TextLoader(tmp_file_path, encoding="utf-8")
                # TextLoader loads the whole file as one document
                docs_from_file = loader.load()
                # Set the source metadata for the text file
                for doc in docs_from_file:
                    doc.metadata["source"] = source_filename
                all_documents.extend(docs_from_file)
            else:
                print(f"Unsupported file type: {source_filename}. Skipping.")
        except Exception as e:
            print(f"Error loading file {source_filename}: {e}")
        finally:
            # Clean up the temporary file
            os.remove(tmp_file_path)
            
    return all_documents


def split_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of documents into smaller chunks.

    Args:
        documents: A list of LangChain Document objects.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of LangChain Document objects, representing the chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Adds the start index of the chunk in the original document
    )
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

# Example usage (for testing this module directly)
if __name__ == '__main__':    
    # Create dummy Streamlit UploadedFile objects (simplified for testing)
    class MockUploadedFile:
        def __init__(self, name, content_bytes):
            self.name = name
            self._content_bytes = content_bytes

        def getvalue(self):
            return self._content_bytes

    # Create a dummy PDF file content (minimal valid PDF)
    dummy_pdf_content = b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000058 00000 n\n0000000112 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF"
    
    dummy_files = [
        MockUploadedFile("test.txt", b"This is a test text document. It has multiple sentences. LangChain is great!"),
        MockUploadedFile("sample.pdf", dummy_pdf_content + b" This is some text on page 1 of PDF. " * 50 +  # page 1
                                     b"\f" +  # Page break
                                     b" This is some text on page 2 of PDF. " * 50) # page 2
    ]

    print("Loading documents...")
    loaded_docs = load_documents(dummy_files)
    print(f"Loaded {len(loaded_docs)} document(s)/page(s).")
    for i, doc in enumerate(loaded_docs):
        print(f"\nDocument {i+1} from '{doc.metadata.get('source', 'N/A')}':")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        # print(f"Content preview: {doc.page_content[:100]}...") # Careful with PDF binary

    if loaded_docs:
        print("\nSplitting documents...")
        # For PDF, PyPDFLoader extracts text, so content is text
        # For the dummy PDF above, PyPDFLoader might not extract text well, 
        # a real PDF with text content is needed for good test of chunking.
        # Let's assume PyPDFLoader successfully extracts text from a real PDF.
        
        # For this direct test, let's use a simpler text-based doc list for splitting demonstration
        simple_docs_for_splitting = [
            Document(page_content="This is the first sentence. This is the second sentence. This is the third. And a fourth.", metadata={"source": "test.txt"}),
            Document(page_content="Another document here. It's fairly short as well.", metadata={"source": "another.txt"})
        ]
        
        chunked_docs = split_documents(simple_docs_for_splitting, chunk_size=50, chunk_overlap=10)
        print(f"Split into {len(chunked_docs)} chunks.")
        for i, chunk in enumerate(chunked_docs):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata.get('source')}, Page: {chunk.metadata.get('page', 'N/A')}, Start Index: {chunk.metadata.get('start_index')}")
            print(f"Content: {chunk.page_content}")
    else:
        print("No documents loaded, skipping splitting.")