from langchain.document_loaders.directory import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain_community.document_loaders import PyPDFLoader

COLLECTION_NAME = "data_langchain"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load_and_split()

vector_store = Qdrant.from_documents(
    documents=documents, embedding=embeddings, collection_name=COLLECTION_NAME
)
