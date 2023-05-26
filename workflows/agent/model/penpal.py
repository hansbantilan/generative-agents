import os

from generative_agents.modeling.interactions import ask_penpal
from generative_agents.modeling.agent_helpers import load_llm, load_pinecone, load_agent
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import load_params

log = logger.init("conversation")

def main():
    llm = load_llm()
    log.info(f"Loading Parameters...")
    params = load_params(
        os.path.join(
            well_known_paths["PARAMS_DIR"],
            "penpal_agent",
            f"default.yaml",
        )
    )

    if params["is_pinecone"]:
        load_pinecone()
    agent = load_agent(params, llm)

    log.info(f"Asking {params['name']} the following question...\n{params['question']}")
    print(ask_penpal(agent, params))


if __name__ == "__main__":
    main()