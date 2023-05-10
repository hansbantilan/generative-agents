from typing import List

from generative_agents.modeling.langchain_agent import GenerativeAgent

user_name = "GPT-Interviewer"


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the user interact with the agent."""
    new_message = f"{user_name} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(
                observation
            )
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1
