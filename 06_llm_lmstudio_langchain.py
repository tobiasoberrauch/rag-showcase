from langchain.llms.openai import OpenAI

llm = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

text = "Was ist die Hauptstadt von Deutschland"
response = llm(text)
print(response)
