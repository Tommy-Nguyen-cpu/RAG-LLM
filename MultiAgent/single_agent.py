from crewai import Agent

class SingleAgent():
    def __init__(self, role, goal, backstory, llm ="ollama/llama3.2", tools = None):
        self.agent = Agent(
            role = role,
            goal = goal,
            backstory = backstory,
            llm = llm,
            tools = tools
        )