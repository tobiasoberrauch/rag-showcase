from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

COLLECTION_NAME = "data_llama_index"

# Initialize the embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = OpenAI(
    temperature=0, 
    api_base='http://localhost:1234/v1',
    api_key='lm-studio'
)

# Initialize the Qdrant client
client = QdrantClient(url="localhost:6333")

vector_store = QdrantVectorStore(collection_name=COLLECTION_NAME, client=client)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine()

response = query_engine.query("Was ist LEAM?")
print(response)
