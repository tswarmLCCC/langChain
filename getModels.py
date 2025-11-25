import requests
import json

# Replace with the actual URL of your Ollama host
# Default is usually http://localhost:11434
LLM_HOST = "http://192.168.1.149:11434"
OLLAMA_HOST_URL = LLM_HOST

def get_ollama_models() -> list:
    """
    Fetches a list of available LLMs from the Ollama host.
    """
    try:
        response = requests.get(f"{OLLAMA_HOST_URL}/api/tags")
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        model_names = []
        if "models" in data:
            for model in data["models"]:
                if "name" in model:
                    model_names.append(model["name"])
        return model_names
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama server at {OLLAMA_HOST_URL}. "
              "Ensure Ollama is running and accessible.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching models: {e}")
        return []

if __name__ == "__main__":
    available_models = get_ollama_models()
    if available_models:
        print("Available Ollama models:")
        for model in available_models:
            print(f"- {model}")
    else:
        print("No Ollama models found or an error occurred.")