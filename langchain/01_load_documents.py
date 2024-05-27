from langchain.document_loaders.directory import DirectoryLoader
from rich import print
from langchain.text_splitter import CharacterTextSplitter

loader = DirectoryLoader("../data")
documents = loader.load()
print(len(documents))


splitter = CharacterTextSplitter(
    chunk_size=1000,  # Maximal erlaubte Zeichen pro Chunk
    chunk_overlap=200,  # Überlappung zwischen den Chunks
    length_function=len  # Funktion, um die Länge der Chunks zu messen
)

# Teile die Dokumente in Chunks auf
chunks = splitter.split_documents(documents)

print(len(chunks))