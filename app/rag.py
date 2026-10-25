from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "enterprise_knowledge"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    pdf_files = list(
        KNOWLEDGE_DIR.rglob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found inside: {KNOWLEDGE_DIR}"
        )

    print("\n" + "=" * 70)
    print("LOADING KNOWLEDGE BASE")
    print("=" * 70)

    print(
        f"Knowledge directory: {KNOWLEDGE_DIR}"
    )

    print(
        f"PDF files found: {len(pdf_files)}"
    )

    for pdf_file in pdf_files:

        print(
            f"\nLoading: {pdf_file.name}"
        )

        loader = PyPDFLoader(
            str(pdf_file)
        )

        pdf_documents = loader.load()

        documents.extend(
            pdf_documents
        )

        print(
            f"Pages loaded: {len(pdf_documents)}"
        )

    print(
        f"\nTotal pages loaded: {len(documents)}"
    )

    return documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=700,

        chunk_overlap=120,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(
    chunks,
    embeddings,
):

    print(
        "\nCreating ChromaDB vector store..."
    )

    vector_store = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        collection_name=COLLECTION_NAME,

        persist_directory=str(
            CHROMA_DIR
        ),
    )

    print(
        "\nChromaDB vector store created successfully."
    )

    print(
        f"Location: {CHROMA_DIR}"
    )

    return vector_store


# ============================================================
# LOAD EXISTING VECTOR STORE
# ============================================================

def load_vector_store(embeddings):

    print(
        "\nLoading existing ChromaDB..."
    )

    vector_store = Chroma(

        collection_name=COLLECTION_NAME,

        embedding_function=embeddings,

        persist_directory=str(
            CHROMA_DIR
        ),
    )

    return vector_store


# ============================================================
# GET VECTOR STORE
# ============================================================

def get_vector_store():

    embeddings = create_embeddings()

    # --------------------------------------------------------
    # Existing ChromaDB
    # --------------------------------------------------------

    if CHROMA_DIR.exists():

        try:

            vector_store = load_vector_store(
                embeddings
            )

            collection = (
                vector_store._collection
            )

            count = collection.count()

            if count > 0:

                print(
                    f"\nExisting ChromaDB loaded."
                )

                print(
                    f"Documents in collection: {count}"
                )

                return vector_store

        except Exception as e:

            print(
                "\nExisting ChromaDB could not be loaded."
            )

            print(
                f"Reason: {e}"
            )

    # --------------------------------------------------------
    # Build new database
    # --------------------------------------------------------

    print(
        "\nNo usable ChromaDB found."
    )

    print(
        "Building vector store from knowledge base..."
    )

    documents = load_documents()

    chunks = split_documents(
        documents
    )

    return create_vector_store(
        chunks,
        embeddings
    )


# ============================================================
# CREATE RETRIEVER
# ============================================================

def create_retriever(vector_store):

    return vector_store.as_retriever(

        search_type="mmr",

        search_kwargs={

            "k": 6,

            "fetch_k": 15,

            "lambda_mult": 0.7,
        },
    )


# ============================================================
# GET RETRIEVER
# ============================================================

def get_retriever():

    vector_store = get_vector_store()

    return create_retriever(
        vector_store
    )


