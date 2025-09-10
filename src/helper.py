from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings





#Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf", #I only wany to load the pdf, not any other file
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents




def filer_to_minimal_docs(documents: List[Document]) -> List[Document]:
    """
    Given a list of document objects, return a new list of document objects containing
    only 'source' in metadata and the original page_content.
    """

    minimal_docs: List[Document] = []
    for doc in documents:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {'source': src}
            )
        )
    return minimal_docs


#split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks



def download_embeddings():
    """
    Download and return the HugginFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings