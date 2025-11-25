
from langchain_ollama import ChatOllama

MODEL_NAME = "deepseek-r1:8b"


class LLMBase:
    llm = None
    def __init__(self):
        if not self.llm:
            self.initialize_llm()

    # You must add base_url here, otherwise it defaults to localhost
    def initialize_llm(self):
        self.llm = ChatOllama(
            model=MODEL_NAME,
            base_url="http://192.168.1.149:11434"
        )
    
    def invoke(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response
    
    def getResponseContent(self, prompt: str):
        response = self.invoke(prompt)
        return response.content

if __name__ == "__main__":
    llm_base = LLMBase()
    
    response = llm_base.invoke(prompt="Hello world")
    print("Test LLM Response:", response)

    print(20*"-")

    response = llm_base.getResponseContent(prompt="Hello world")

    print("Response from LLM:", response)