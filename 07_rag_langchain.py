from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient

COLLECTION_NAME = "data_langchain"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url="localhost:6333")

vector_store = Qdrant(
    client=client, collection_name=COLLECTION_NAME, embeddings=embeddings
)

prompt_template = """
Alle Antworten werden ausschließlich aus LEAM-Handbuch abgeleitet. 
Falls Ihre Frage Informationen benötigt, die über den Rahmen dieses Dokuments hinausgehen, 
werde ich darauf hinweisen, dass eine Antwort darauf nicht im Handbuch enthalten ist. 
Bitte achten Sie darauf, dass Ihre Fragen direkt auf den Inhalt des LEAM-Handbuch bezogen sind.

{context}

Frage: {question}
Antwort hier:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {
    "prompt": PROMPT,
    "verbose": True
}

llm = OpenAI(
    model="TheBloke/em_german_leo_mistral-GGUF", 
    temperature=0, base_url='http://localhost:1234/v1',
    api_key='lm-studio'
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True,
)

query = "Was ist LEAM?"
print(qa.invoke(query))
