import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool # A pre-built tool

# --- Setup ---
MODEL_NAME = "llama3.2:latest"

# --- 1. Define Tools ---
# An agent uses tools to interact with the outside world.
# The `@tool` decorator makes it easy to convert a function into a tool.

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    print(f"--- Tool Used: get_word_length(word='{word}') ---")
    return len(word)

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    print(f"--- Tool Used: multiply(a={a}, b={b}) ---")
    return a * b

# You can also import pre-built tools
# This tool allows the agent to execute Python code.
# Be careful: This is powerful but can be a security risk!
python_repl_tool = PythonREPLTool()

# List of tools the agent can use
tools = [get_word_length, multiply, python_repl_tool]

# --- 2. Initialize the Model ---
print("Initializing ChatOllama model...")
llm = ChatOllama(model=MODEL_NAME, temperature=0) # Low temp for better tool use

# --- 3. Create the Agent Prompt ---
# This is a special prompt that tells the agent HOW to reason.
# It uses the ReAct (Reasoning and Acting) framework.
# The prompt is pulled from the LangChain Hub to ensure it's formatted correctly.
# Note: This is a complex string, so using the pre-built one is easiest.
react_prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(react_prompt_template)

# --- 4. Create the Agent ---
# This binds the LLM, the prompt, and the tools together.
print("Creating ReAct agent...")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# --- 5. Create the Agent Executor ---
# The executor is what actually RUNS the agent's thought loop.
print("Creating agent executor...")
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # Set to True to see the agent's "thoughts"
)

# --- 6. Run the Agent ---
print("\n--- Running Agent (Example 1) ---")
try:
    query1 = "What is the length of the word 'onomatopoeia' and what is 12 * 12?"
    response1 = agent_executor.invoke({
        "input": query1
    })
    print("\n" + "="*30)
    print(f"Query: {query1}")
    print(f"\nFinal Answer: {response1['output']}")
    print("="*30)

    print("\n--- Running Agent (Example 2) ---")
    query2 = "what are the first 5 prime numbers?"
    response2 = agent_executor.invoke({
        "input": query2
    })
    print("\n" + "="*30)
    print(f"Query: {query2}")
    print(f"\nFinal Answer: {response2['output']}")
    print("="*30)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print(f"Please ensure Ollama is running and the model '{MODEL_NAME}' is available.")

print("\nScript finished.")