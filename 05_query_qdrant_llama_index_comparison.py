from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.vector_stores.types import VectorStoreQueryMode

COLLECTION_NAME = "data_llama_index"

# Set up the embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.chunk_size = 512

client = QdrantClient("http://localhost:6333")

# Create a vector store
vector_store = QdrantVectorStore(
    client=client, 
    collection_name=COLLECTION_NAME, 
    enable_hybrid=True,
    batch_size=20
)

# Create a VectorStoreIndex
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_modes = [mode for mode in VectorStoreQueryMode]

for mode in query_modes:
    retriever = index.as_retriever(vector_store_query_mode=mode.value)
    documents = retriever.retrieve("Was ist LEAM?")
    print(f"Documents for query mode {mode.value}:", documents)