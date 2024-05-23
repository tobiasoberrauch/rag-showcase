from langchain.document_loaders.directory import DirectoryLoader

loader = DirectoryLoader("./data")
documents = loader.load()
print(documents)