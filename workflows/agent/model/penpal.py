import argparse
import os

from generative_agents.modeling.agent_helpers import load_agent, load_llm, load_pinecone
from generative_agents.modeling.interactions import talk_to_penpal
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import load_params

log = logger.init("Pen Pal")


def main(
    llm_type: str,
    temperature: float,
) -> None:
    llm = load_llm(llm_type, temperature)
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

    log.info(f"Talking to {params['name']}...\n{params['user_input']}")
    print(talk_to_penpal(agent, params["user_input"], params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pen Pal")
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
