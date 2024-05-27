from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader("./data")

documents = reader.load_data()
print(documents)