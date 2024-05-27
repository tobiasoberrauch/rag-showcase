import time
import csv
from langchain.document_loaders.directory import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient

def benchmark(chunk_size, chunk_overlap, splitter_cls):
    collection_name = f"data_langchain_{splitter_cls.__name__}_chunk{chunk_size}_overlap{chunk_overlap}"
    print(f"Benchmarking with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, collection_name={collection_name}, splitter={splitter_cls.__name__}")

    # Start timer for document loading and splitting
    start_time = time.time()

    # Define the text splitter with given parameters
    splitter = splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load and split the documents
    loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load_and_split(text_splitter=splitter)

    load_and_split_time = time.time() - start_time
    print(f"Time taken to load and split documents: {load_and_split_time:.2f} seconds")

    # Start timer for vector store creation
    start_time = time.time()

    # Define the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create the vector store
    vector_store = Qdrant.from_documents(
        documents=documents, embedding=embeddings, collection_name=collection_name
    )

    vector_store_time = time.time() - start_time
    print(f"Time taken to create vector store: {vector_store_time:.2f} seconds")

    # Define the LLM and QA chain
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

    llm = OpenAI(
        model="TheBloke/em_german_leo_mistral-GGUF", 
        temperature=0, base_url='http://localhost:1234/v1',
        api_key='lm-studio'
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT, "verbose": True},
        return_source_documents=True,
    )

    # Start timer for query
    start_time = time.time()

    query = "Was ist LEAM?"
    result = qa.invoke(query)

    query_time = time.time() - start_time
    print(f"Time taken to answer query: {query_time:.2f} seconds")

    # Clean up the collection
    client = QdrantClient(url="http://localhost:6333")
    client.delete_collection(collection_name)

    return load_and_split_time, vector_store_time, query_time, result['result']

# Define the chunk sizes, overlaps, and splitters to test
chunk_sizes = [500, 1000, 1500]
chunk_overlaps = [0, 100, 200]
splitters = [CharacterTextSplitter, RecursiveCharacterTextSplitter]

# Run the benchmarks
results = []

for splitter_cls in splitters:
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            result = benchmark(chunk_size, chunk_overlap, splitter_cls)
            results.append((splitter_cls.__name__, chunk_size, chunk_overlap, *result))

# Save results to CSV
csv_filename = "benchmark_results.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Splitter", "Chunk Size", "Chunk Overlap", "Load and Split Time", "Vector Store Time", "Query Time", "QA Result"])
    for result in results:
        writer.writerow(result)

print(f"Results have been saved to {csv_filename}")

# Print the results
for result in results:
    splitter_name, chunk_size, chunk_overlap, load_and_split_time, vector_store_time, query_time, qa_result = result
    print(f"Splitter: {splitter_name}, Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")
    print(f"Load and Split Time: {load_and_split_time:.2f} seconds")
    print(f"Vector Store Time: {vector_store_time:.2f} seconds")
    print(f"Query Time: {query_time:.2f} seconds")
    print(f"QA Result: {qa_result}")
    print("-" * 40)
