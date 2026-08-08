import os
import uuid
import chromadb
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

DEFAULT_DOC_PATH = "docs"
DEFAULT_PERSIST_DIR = "db/chroma_db"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB total per upload batch

_embedding_model = None


def get_embedding_model():
    """Reuse a single embedding model instance across calls instead of reloading it each time."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
    return _embedding_model


def load_documents(doc_path=DEFAULT_DOC_PATH):
    print(f"Loading documents from {doc_path} location....")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"The directory {doc_path} does not exist.")

    loader = DirectoryLoader(
        path=doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True,
        },
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file found in the path {doc_path}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=100):
    print("\nSplitting documents into chunks....")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Total Number of Chunks: {len(chunks)}")
    return chunks


def create_vector_store(chunks, persist_directory=DEFAULT_PERSIST_DIR):
    print("\nCreating embeddings and storing in Chroma DB")

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("\nFinished creating vector store")
    return vectorstore


def build_vectorstore_if_missing(doc_path=DEFAULT_DOC_PATH, persist_directory=DEFAULT_PERSIST_DIR):
    """
    Builds the Chroma vector store only if it doesn't already exist on disk.
    Needed because Hugging Face Spaces free tier has non-persistent disk —
    this rebuilds it fresh each time the Space restarts/cold-starts.
    """
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Vector store already exists at {persist_directory}, skipping rebuild.")
        return

    documents = load_documents(doc_path=doc_path)
    chunks = split_documents(documents)
    create_vector_store(chunks, persist_directory=persist_directory)


def enforce_upload_size_limit(file_paths, max_bytes=MAX_UPLOAD_BYTES):
    """
    Raises a friendly error if the combined size of all uploaded files
    exceeds the allowed total (default 100MB).
    """
    total_bytes = sum(os.path.getsize(p) for p in file_paths if os.path.exists(p))
    if total_bytes > max_bytes:
        total_mb = total_bytes / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(
            f"Total upload size is {total_mb:.1f}MB, which is over the {limit_mb:.0f}MB limit. "
            f"Please upload fewer or smaller files."
        )


def _load_xlsx(path):
    """
    Reads every sheet of an Excel file with pandas and turns each sheet into
    one Document (rather than one Document per row, to keep chunk counts sane).
    """
    sheets = pd.read_excel(path, sheet_name=None)
    docs = []
    for sheet_name, df in sheets.items():
        text = df.to_csv(index=False)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path, "sheet": sheet_name},
            )
        )
    return docs


def load_uploaded_files(file_paths):
    """
    Loads .txt, .pdf, .docx, .csv, and .xlsx/.xls files given a list of file paths
    (e.g. from a Gradio upload). Files that fail to load or have an unsupported
    extension are skipped individually rather than failing the whole batch.
    Returns (documents, skipped) where skipped is a list of (path, reason) tuples.
    """
    documents = []
    skipped = []

    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".txt":
                documents.extend(TextLoader(path, encoding="utf-8", autodetect_encoding=True).load())
            elif ext == ".pdf":
                documents.extend(PyPDFLoader(path).load())
            elif ext == ".docx":
                documents.extend(Docx2txtLoader(path).load())
            elif ext == ".csv":
                documents.extend(CSVLoader(path).load())
            elif ext in (".xlsx", ".xls"):
                documents.extend(_load_xlsx(path))
            else:
                skipped.append((path, f"unsupported file type '{ext}'"))
        except Exception as e:
            skipped.append((path, str(e)))

    if not documents:
        raise ValueError(
            "No files could be read. Supported types are .txt, .pdf, .docx, .csv, .xlsx, .xls."
        )

    return documents, skipped


def build_vectorstore_from_uploads(file_paths, chunk_size=800, chunk_overlap=0):
    """
    Builds a fresh, in-memory (non-persisted) Chroma vector store from user-uploaded files
    of mixed types (pdf, txt, docx, csv, xlsx) processed together in one batch.
    Each call returns a brand-new store, so each visitor's session gets its own private
    vector store instead of writing to shared disk or mixing with other visitors' files.
    Returns (vectorstore, skipped) where skipped lists any files that couldn't be read.
    """
    enforce_upload_size_limit(file_paths)

    documents, skipped = load_uploaded_files(file_paths)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embedding_model = get_embedding_model()

    # Create a brand-new, isolated ephemeral client + unique collection name every time.
    # Without this, Chroma can reuse a shared default in-memory collection across calls,
    # so old files' embeddings linger even after you "removed" that file and uploaded a new one.
    client = chromadb.EphemeralClient()
    collection_name = f"session-{uuid.uuid4().hex}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
        # no persist_directory -> stays in memory, never written to disk
    )
    print(f"Built in-memory vector store from {len(file_paths)} uploaded file(s), {len(chunks)} chunks.")
    return vectorstore, skipped


def main():
    documents = load_documents(doc_path=DEFAULT_DOC_PATH)
    chunks = split_documents(documents)
    create_vector_store(chunks)


if __name__ == "__main__":
    main()
