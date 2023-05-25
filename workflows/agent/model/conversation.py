import os

import pinecone
from langchain.chat_models import ChatOpenAI

from generative_agents.modeling.interactions import interview_agent, run_conversation
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
        "langchain_agent",
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

log.info(f"Starting a conversation between agents...\n{params['conversation_starter']}")
run_conversation(agents, params["conversation_starter"])

log.info(
    f"Interviewing each agent after the conversation...\n{params['interview_question']}"
)
for agent in agents:
    print(interview_agent(agent, params["interview_question"]))
