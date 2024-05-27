from datasets import Dataset 
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from ragas.evaluation import evaluate
from ragas.metrics import answer_relevancy, answer_similarity, answer_correctness

# Configuration
COLLECTION_NAME = "data_langchain"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "TheBloke/em_german_leo_mistral-GGUF"
LLM_BASE_URL = 'http://localhost:1234/v1'
API_KEY = 'lm-studio'
QDRANT_URL = "localhost:6333"

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize Qdrant Client
client = QdrantClient(url=QDRANT_URL)

# Initialize Qdrant Vector Store
vector_store = Qdrant(
    client=client, collection_name=COLLECTION_NAME, embeddings=embeddings
)

# Define the Prompt Template
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

# Chain Configuration
chain_type_kwargs = {
    "prompt": PROMPT,
    "verbose": True
}

# Initialize the OpenAI LLM
llm = OpenAI(
    model=LLM_MODEL, 
    temperature=0, 
    base_url=LLM_BASE_URL,
    api_key=API_KEY
)

# Define the RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True,
)

# Query the Chain
query = "Was ist LEAM?"
response = qa.invoke(query)
generated_answer = response['result']

# Debugging: Print the generated answer and source documents
print("Generated Answer:", generated_answer)
print("Source Documents:", response['source_documents'])

# Extract plain text from the context documents
context_texts = [doc.page_content for doc in response['source_documents']]

# Debugging: Print the context texts
print("Context Texts:", context_texts)

# Define the evaluation dataset
eval_dataset = {
    "question": ["Was ist LEAM?"],
    "contexts": [context_texts],
    "answer": [generated_answer],
    "ground_truth": ["LEAM, das für 'Large European AI Models' steht, ist eine Initiative des KI Bundesverbands (German AI Association). Das Hauptziel von LEAM ist die Förderung der Entwicklung großer KI-Modelle in Europa, um sicherzustellen, dass Europa im globalen KI-Bereich wettbewerbsfähig bleibt."]
}
dataset = Dataset.from_dict(eval_dataset)

# Debugging: Print the evaluation dataset
print("Evaluation Dataset:", dataset)

# Evaluate the RAG pipeline using RAGAS
try:
    score = evaluate(
        dataset,
        metrics=[
            answer_similarity,
            answer_correctness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True  # Ensure exceptions are raised for better debugging
    )
    print("Evaluation Score:", score)
except Exception as e:
    print("Evaluation failed:", str(e))

    # Additional debug information
    import traceback
    traceback.print_exc()
