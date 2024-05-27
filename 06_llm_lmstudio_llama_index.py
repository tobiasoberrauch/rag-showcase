from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="http://localhost:1234/v1",
    api_key="lm-studio"
)

response = llm.complete("Was ist die Hauptstadt von Deutschland")
print(response)