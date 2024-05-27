from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3")

response = llm.complete("Was ist die Hauptstadt von Deutschland")
print(response)