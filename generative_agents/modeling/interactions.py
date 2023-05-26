from typing import List

from generative_agents.modeling.langchain_agent import GenerativeAgent
from generative_agents.utility import logger

user_name = "User"

log = logger.init("interactions.py")

def start_conversation_w_penpal(agent: GenerativeAgent, params: dict, location: str) -> str:
    log.info(location)
    observation = f"You are meeting {user_name} at a location delimited by triple backticks. ```{location}```"
    # TODO get something more relevant to the location, context
    return agent.get_penpal_answer(observation, params)

def talk_to_penpal(agent: GenerativeAgent, user_input, params: dict) -> str:
    observation = f"{user_name} says {user_input}"
    return agent.get_penpal_answer(observation, params)


def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the user interact with the agent."""
    observation = f"{user_name} says {message}"
    return agent.generate_dialogue_response(observation)[1]


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
