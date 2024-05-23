from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = SimpleDirectoryReader("./data").load_data()

# Create a vector store
client = QdrantClient("http://localhost:6333", api_key=None)
vector_store = QdrantVectorStore(client=client, collection_name="data_llama_index")

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create a GPTVectorStoreIndex
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
