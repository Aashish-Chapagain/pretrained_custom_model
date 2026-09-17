"""Example showing how to load and use MiniLLM in any external project."""
from minillm import MiniLLMRuntime, load_model

# 1. Load the model from this directory
llm = load_model(".")

# 2. Single generation
prompt = "The solar system"
output = llm.generate(prompt, max_new_tokens=60, temperature=0.35)
print("Generated:")
print(output)

# 3. Or start interactive chat:
# llm.chat()
