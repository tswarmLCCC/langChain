import os
# This environment variable fixes a known dependency conflict with Keras 3.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough





# --- Setup ---
MODEL_NAME = "llama3.2:latest"
MODEL_NAME = "deepseek-r1:8b"

LLM_HOST = "http://localhost:11434"
#LLM_HOST = "http://192.168.1.149:11434"
#LLM_HOST = os.getenv("LLM_HOST", LLM_HOST)

# Define the ethical dilemma
dilemma = """
The AI Dilemma: A self-driving car's brakes have failed. 
It is on a path to crash into a barrier, which will kill the ONE (1) passenger inside. 
The car has the option to swerve into a crowd of FIVE (5) pedestrians, killing all of them but saving the passenger.
What is the most ethical action for the car to take?
"""

# --- Agent 1: Kantian ---
kantian_system_prompt = """
You are a philosopher adhering strictly to Kantian deontology.
Your reasoning is based on the Categorical Imperative: "Act only according to that maxim whereby you can, at the same time, will that it should become a universal law."
This means:
1.  **You do not care about consequences.** You care about the rightness or wrongness of the action itself.
2.  **You must not "use" people as a "mere means" to an end.** Killing the 5 pedestrians to save the 1 passenger is using them as a tool, which is morally forbidden.
3.  An action is either a "perfect duty" (like "do not kill an innocent person") or it is not.

You will debate another philosopher about the correct action.
Your first argument should state your position on the dilemma.
"""

# --- Agent 2: Utilitarian ---
utilitarian_system_prompt = """
You are a philosopher adhering strictly to Act Utilitarianism.
Your reasoning is based on the Principle of Utility: "The most ethical action is the one that maximizes overall happiness and minimizes overall suffering."
This means:
1.  **You ONLY care about consequences.** The intention behind the action does not matter, only the outcome.
2.  **You are impartial.** The passenger's life is not more or less valuable than a pedestrian's life.
3.  **You must perform a 'moral calculus'.** You will weigh the outcomes:
    - Option 1 (Do not swerve): 1 death.
    - Option 2 (Swerve): 5 deaths.
    - 1 death is less suffering than 5 deaths.

You will debate another philosopher about the correct action.
Your first argument should state your position on the dilemma.
"""

# --- Debate Setup ---
llm = ChatOllama(model=MODEL_NAME, temperature=0.7, host=LLM_HOST)
print("Initialized ChatOllama model for debate.", llm)


# The debate chain for each agent
def create_debate_chain(system_prompt):
    """Creates a runnable chain for an agent with a specific persona."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "The debate topic is: {topic}"),
        ("ai", "I am ready to begin the debate."),
        ("human", "{argument}")
    ])
    return (
        {"topic": RunnablePassthrough(), "argument": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Create the two agent chains
kantian_agent_chain = create_debate_chain(kantian_system_prompt)
utilitarian_agent_chain = create_debate_chain(utilitarian_system_prompt)

# --- Run the Debate ---
def run_debate(topic, rounds=3):
    """
    Runs a multi-turn debate between the two agents.
    """
    print(f"Topic: {topic}\n")
    print("--- DEBATE START ---")

    # Start the debate
    # The Kantian agent makes the first statement on the topic.
    kantian_argument = kantian_agent_chain.invoke(
        {"topic": topic, "argument": "What is your initial position on this dilemma?"}
    )
    print(f"Kantian [Round 1]:\n{kantian_argument}\n")
    
    # Initialize the argument for the loop
    current_argument = kantian_argument

    for i in range(rounds):
        # Utilitarian responds
        utilitarian_argument = utilitarian_agent_chain.invoke(
            {"topic": topic, "argument": current_argument}
        )
        print(f"Utilitarian [Round {i+1}]:\n{utilitarian_argument}\n")
        
        # Kantian responds
        kantian_argument = kantian_agent_chain.invoke(
            {"topic": topic, "argument": utilitarian_argument}
        )
        print(f"Kantian [Round {i+2}]:\n{kantian_argument}\n")
        
        # The Kantian's last argument becomes the next input for the Utilitarian
        current_argument = kantian_argument

    print("--- DEBATE END ---")

if __name__ == "__main__":
    run_debate(dilemma, rounds=2)