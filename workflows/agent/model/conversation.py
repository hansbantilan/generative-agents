import argparse
import os

from generative_agents.modeling.agent_helpers import load_llm, load_pinecone
from generative_agents.modeling.interactions import interview_agent, run_conversation
from generative_agents.modeling.langchain_agent import GenerativeAgent
from generative_agents.modeling.memory import create_new_memory_retriever
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import load_params

log = logger.init("Interview")


def main(
    llm_type: str,
    temperature: float,
) -> None:
    llm = load_llm(llm_type, temperature)
    log.info(f"Loading Parameters...")
    params = load_params(
        os.path.join(
            well_known_paths["PARAMS_DIR"],
            "langchain_agent",
            f"default.yaml",
        )
    )

    if params["is_pinecone"]:
        load_pinecone()

    agents = list()
    for agent_key in params["agents"].keys():
        _params = params["agents"][agent_key]

        log.info(f"Instantiating agent: {_params['name']}...")
        agent = GenerativeAgent(
            name=_params["name"],
            backstory=_params["backstory"],
            traits=_params["traits"],
            status=_params["status"],
            memory_retriever=create_new_memory_retriever(params["is_pinecone"]),
            llm=llm,
            daily_summaries=_params["daily_summaries"],
            reflection_threshold=_params["reflection_threshold"],
        )
        agents.append(agent)

        log.info(f"  Adding {_params['name']}'s memories...")
        for memory in _params["memories"]:
            agent.add_memory(memory)

    log.info(
        f"Starting a conversation between agents...\n{params['conversation_starter']}"
    )
    run_conversation(agents, params["conversation_starter"])

    log.info(
        f"Interviewing each agent after the conversation...\n{params['interview_question']}"
    )
    for agent in agents:
        print(interview_agent(agent, params["interview_question"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interview")
    parser.add_argument(
        "--llmType",
        default="GPT-3.5-turbo",
        action="store",
        dest="llm_type",
        choices=["GPT-3.5-turbo", "GPT-4", "Claude-v1"],
        help="One of {GPT-3.5-turbo, GPT-4, Claude-v1}",
    )
    parser.add_argument(
        "--temperature",
        action="store",
        default=0.9,
        dest="temperature",
        type=float,
        help="A non-negative float that tunes the degree of randomness in generation",
    )

    args = parser.parse_args()
    print(vars(args))
    for k, v in vars(args).items():
        exec(f"{k} = '{v}'")
    main(**vars(args))
