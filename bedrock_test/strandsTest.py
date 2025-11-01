# strandsTest.py
from strands import Agent, tool
from strands_tools import calculator, current_time

# Define your own custom tool
@tool
def letter_counter(word: str, letter: str) -> int:
    """Counts how many times a given letter appears in a word."""
    if not isinstance(word, str) or not isinstance(letter, str):
        return 0
    if len(letter) != 1:
        raise ValueError("The 'letter' parameter must be a single character")
    return word.lower().count(letter.lower())

# Create the agent with a list of available tools
agent = Agent(tools=[calculator, current_time, letter_counter])

# Message for the agent to handle
message = """
I have three requests:
1. What is the time right now?
2. Calculate 3111696 / 74088
3. Tell me how many letter R's are in the word "strawberry".
"""

# Ask the agent to handle the message
response = agent(message)
print(response)
