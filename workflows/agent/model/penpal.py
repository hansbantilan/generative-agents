import os

import pinecone
from langchain.chat_models import ChatOpenAI

from generative_agents.modeling.interactions import talk_to_penpal
from generative_agents.modeling.langchain_agent import GenerativeAgent
from generative_agents.modeling.memory import create_new_memory_retriever
from generative_agents.utility import logger, well_known_paths
from generative_agents.utility.utility import load_params

log = logger.init("conversation")


log.info(f"Loading LLM...")
llm = ChatOpenAI(max_tokens=1000)

log.info(f"Loading Parameters...")
params = load_params(
    os.path.join(
        well_known_paths["PARAMS_DIR"],
        "penpal_agent",
        f"default.yaml",
    )
)

if params["is_pinecone"]:
    log.info(f"Loading Pinecone Vector Database...")
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )
    if "generative-agents-index" not in pinecone.list_indexes():
        log.info(f"Creating generative-agents-index in Pinecone Vector Database...")
        pinecone.create_index("generative-agents-index", dimension=1536)

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

log.info(f"  Adding {params['name']}'s memories...")
for memory in params["memories"]:
    agent.add_memory(memory)

log.info(f"Talking to {params['name']}...\n{params['user_input']}")
print(talk_to_penpal(agent, params))
