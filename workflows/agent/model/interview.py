import argparse
import os

import pandas as pd

from generative_agents.modeling.agent_helpers import load_llm, load_pinecone
from generative_agents.modeling.interactions import interview_agent
from generative_agents.modeling.langchain_agent import GenerativeAgent
from generative_agents.modeling.memory import create_new_memory_retriever
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import load_params

log = logger.init("Conversation")


def main(
    llm_type: str,
    temperature: float,
) -> None:
    llm = load_llm(llm_type, temperature)
    log.info(f"Loading Parameters...")
    params = load_params(
        os.path.join(
            well_known_paths["PARAMS_DIR"],
            "interview_agent",
            f"default.yaml",
        )
    )

    if params["is_pinecone"]:
        load_pinecone()

    log.info(f"Instantiating agent: {params['name']}...")
    agent = GenerativeAgent(
        name=params["name"],
        backstory=params["backstory"],
        traits=params["traits"],
        status=params["status"],
        memory_retriever=create_new_memory_retriever(params["is_pinecone"]),
        llm=llm,
        daily_summaries=params["daily_summaries"],
        reflection_threshold=params["reflection_threshold"],
    )
    log.info(f"  Loading {params['name']}'s memories...")
    if params["data_source"] == "local":
        file_path = os.path.join(well_known_paths["DATASETS_DIR"], params["file_name"])
        data = pd.read_csv(file_path)
        memories = list()
        for string in data[params["column_name"]].to_list():
            memories.append(f"{params['name']} remembers saying: " + string)
    elif params["data_source"] == "params":
        memories = params["memories"]
    else:
        raise NotImplementedError("Check data_source, must be in {'local','params'}...")
    log.info(f"  Adding {params['name']}'s memories...")
    for memory in memories:
        agent.add_memory(memory)

    log.info(f"Talking to {params['name']}...\n{params['user_input']}")
    print(interview_agent(agent, params["user_input"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversation")
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
