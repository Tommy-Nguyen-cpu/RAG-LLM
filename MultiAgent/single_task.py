from crewai import Task

class SingleTask():
    def __init__(self, description, expected_output, agent, context = None):
        self.task = Task(
            description = description,
            expected_output = expected_output,
            agent = agent,
            context = context
        )