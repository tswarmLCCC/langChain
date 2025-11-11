import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Setup ---
# 1. Make sure Ollama is running
#    (e.g., `ollama run llama3`)
# 2. This script assumes you have a model named 'llama3'
#    You can change this to any model you have pulled in Ollama.
MODEL_NAME = "llama3.2:latest"

# --- 1. Initialize the Model ---
# Point to the local Ollama instance
print("Initializing ChatOllama model...")
llm = ChatOllama(model=MODEL_NAME)

# --- 2. Create a Prompt Template ---
# This template will guide the AI's response.
print("Creating prompt template...")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who provides concise answers."),
    ("human", "What is the primary benefit of using {topic} in software development?")
])

# --- 3. Create a Simple Chain (LCEL) ---
# We use LangChain Expression Language (LCEL) to pipe components together.
# This simple chain will:
# 1. Take the user's input (a dictionary with "topic").
# 2. Format the prompt.
# 3. Send the formatted prompt to the LLM.
# 4. Get the LLM's response.
# 5. Parse the output into a simple string.
print("Creating simple LLM chain...")
chain = prompt | llm | StrOutputParser()

# --- 4. Run the Chain ---
print("Running chain...")
topic = "Docker containers"
try:
    response = chain.invoke({"topic": topic})
    
    print("\n" + "="*30)
    print(f"Query: What is the primary benefit of using {topic} in software development?")
    print("\nResponse:")
    print(response)
    print("="*30)

except Exception as e:
    print(f"\nAn error occurred:")
    print(f"Error: {e}")
    print("\nPlease ensure Ollama is running and the model '{MODEL_NAME}' is available.")
    print("You can pull the model with: 'ollama pull {MODEL_NAME}'")

print("\nScript finished.")