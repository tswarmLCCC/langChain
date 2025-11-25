import os
#from langchain_community.chat_models import ChatOllama
import utils
import llm_base
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# --- Setup ---
#MODEL_NAME = "llama3.2:latest"
MAX_TURNS = 4

# --- 1. Define the Agents ---

class DebateAgent:
    """A simple agent class for a debate."""
    def __init__(self, name: str, persona: str, llm):
        self.name = name
        self.persona = persona
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.persona + "\n\nYour name is {name}. You are in a debate. Respond to the last statement from the other debater. Be concise."),
            ("human", "{input}")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        self.history = [SystemMessage(content=self.persona)]

    def invoke(self, input_message: str) -> str:
        """Generate a response based on the input message."""
        print(f"--- {self.name} is thinking... ---")
        
        # Add the new message to history (as if it's from a human)
        self.history.append(HumanMessage(content=input_message))
        
        # Create the chain to invoke
        # We pass the full history to maintain context
        prompt = ChatPromptTemplate.from_messages(self.history)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({})
        
        # Add its own response to its history
        self.history.append(SystemMessage(content=response)) # Use SystemMessage for its own response
        
        return response

# --- 2. Initialize Model and Agents ---
print("Initializing ChatOllama model...")
#llm = ChatOllama(model=MODEL_NAME, temperature=0.7)

llm_base_instance = llm_base.LLMBase()
llm = llm_base_instance.get_llm()
print("Initialized ChatOllama model for debate.", llm)

print("Creating agents...")
debate_topic = "Is social media more harmful or beneficial for society?"

# Agent 1: Pro-social media
agent_pro = DebateAgent(
    name="Optimist",
    persona="You are a tech optimist. You believe social media is a powerful tool for connection, education, and positive change. You focus on its benefits.",
    llm=llm
)

# Agent 2: Anti-social media
agent_con = DebateAgent(
    name="Pessimist",
    persona="You are a social critic. You believe social media is detrimental, causing anxiety, misinformation, and echo chambers. You focus on its harms.",
    llm=llm
)

# --- 3. Run the Debate ---
print(f"\n--- Starting Debate ---")
print(f"Topic: {debate_topic}")
print("="*30)

# Start the debate
current_statement = f"Let's begin the debate. The topic is: {debate_topic}"
print(f"Moderator: {current_statement}\n")

# Loop for a few turns
for turn in range(MAX_TURNS):
    # Agent Pro's turn
    response_pro = agent_pro.invoke(current_statement)
    print(f"{utils.GREEN}{agent_pro.name}: {response_pro}\n{utils.RESET}")
    
    # This becomes the input for the next agent
    current_statement = response_pro
    
    # Agent Con's turn
    response_con = agent_con.invoke(current_statement)
    print(f"{agent_con.name}: {response_con}\n")
    
    # This becomes the input for the next agent
    current_statement = response_con

print("="*30)
print("Debate finished.")
print("\n--- Final Histories (for context) ---")
print(f"\n{agent_pro.name}'s History:\n{agent_pro.history}")
print(f"\n{agent_con.name}'s History:\n{agent_con.history}")