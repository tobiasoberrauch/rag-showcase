from langchain.document_loaders.directory import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

COLLECTION_NAME = "data_langchain"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Definiere den Textsplitter mit den Standardeinstellungen
splitter = CharacterTextSplitter()

# Zeige die aktuellen Einstellungen des Textsplitters an
print(f"Chunk Size: {splitter._chunk_size}")
print(f"Chunk Overlap: {splitter._chunk_overlap}")

# Lade und splitte die Dokumente mit den definierten Einstellungen
loader = DirectoryLoader("../data", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load_and_split(text_splitter=splitter)

vector_store = Qdrant.from_documents(
    documents=documents, embedding=embeddings, collection_name=COLLECTION_NAME
)