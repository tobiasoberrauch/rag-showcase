from langchain.llms.ollama import Ollama

llm = Ollama(model="llama3")

text = "Was ist die Hauptstadt von Deutschland?"
print(llm.invoke(text))
