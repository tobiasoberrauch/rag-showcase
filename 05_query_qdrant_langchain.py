from typing import List

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table


def create_table(title: str, documents: List[Document]):
    table = Table(title=title, show_lines=True)
    table.add_column("page")
    table.add_column("page_content")

    for document in documents:
        table.add_row(str(document.metadata["page"]), document.page_content)

    return table


COLLECTION_NAME = "data_langchain"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url="localhost:6333")

vector_store = Qdrant(
    client=client, collection_name=COLLECTION_NAME, embeddings=embeddings
)
# similarity_search
similar_documents = vector_store.search("Was ist LEAM?", "similarity")

# max_marginal_relevance_search
max_marginal_documents = vector_store.search("Was ist LEAM?", "mmr")

console = Console()
console.print(create_table("similar_documents", similar_documents))
console.print(create_table("max_marginal_documents", max_marginal_documents))