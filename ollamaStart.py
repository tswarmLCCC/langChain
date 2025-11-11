import os
from langchain.llms import Ollama
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

#!/usr/bin/env python3
"""
starter_app.py

Minimal starter app showing Ollama + LangChain usage.

Requirements:
    pip install langchain

By default this connects to an Ollama daemon at http://localhost:11434.
Set OLLAMA_URL and OLLAMA_MODEL environment variables to change.
"""



def make_llm():
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")  # adjust to a model available locally
    return Ollama(model=model, base_url=base_url)


def demo_prompt_chain(llm):
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a concise, friendly 3-sentence summary about {topic}."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    sample = "LangChain + Ollama integration"
    out = chain.run(sample)
    print("\n=== Prompt chain example ===")
    print("Input:", sample)
    print("Output:", out.strip())


def run_conversation_cli(llm):
    memory = ConversationBufferMemory()
    conv = ConversationChain(llm=llm, memory=memory, verbose=False)
    print("\n=== Conversation CLI ===")
    print("Type messages. Enter 'exit' or Ctrl-C to quit.\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ("exit", "quit"):
                break
            reply = conv.predict(input=user)
            print("Assistant:", reply.strip())
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


def main():
    llm = make_llm()
    demo_prompt_chain(llm)
    run_conversation_cli(llm)


if __name__ == "__main__":
    main()