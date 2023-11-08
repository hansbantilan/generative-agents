import os

import pinecone
from langchain.chat_models import ChatAnthropic, ChatOpenAI

from generative_agents.modeling.langchain_agent import GenerativeAgent
from generative_agents.modeling.memory import create_new_memory_retriever
from generative_agents.utility import logger

log = logger.init("agent helpers")


def load_llm(llm_type: str, temperature: int):
    log.info(f"Loading LLM: {llm_type}...")
    if llm_type == "GPT-3.5-turbo":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
    elif llm_type == "GPT-4-turbo":
        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview", temperature=temperature, max_tokens=1000
        )
    elif llm_type == "Claude-v1":
        llm = ChatAnthropic(model="claude-v1", temperature=temperature)
    else:
        raise NotImplementedError("This LLM type is not supported...")
    return llm


def load_pinecone():
    log.info(f"Loading Pinecone Vector Database...")
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )
    if "generative-agents-index" not in pinecone.list_indexes():
        log.info(f"Creating generative-agents-index in Pinecone Vector Database...")
        pinecone.create_index("generative-agents-index", dimension=1536)


def load_agent(params, llm):
    name = params["name"]
    log.info(f"Instantiating agent: {name}...")
    backstory = params["backstory"].replace("EF-PenPal", name)
    traits = params["traits"].replace("EF-PenPal", name)
    status = params["status"].replace("EF-PenPal", name)
    daily_summaries = [
        summary.replace("EF-PenPal", name) for summary in params["daily_summaries"]
    ]
    agent = GenerativeAgent(
        name=name,
        backstory=backstory,
        traits=traits,
        status=status,
        memory_retriever=create_new_memory_retriever(params["is_pinecone"]),
        llm=llm,
        daily_summaries=daily_summaries,
        reflection_threshold=params["reflection_threshold"],
    )

    memories = [m.replace("EF-PenPal", name) for m in params["memories"]]
    for memory in memories:
        log.info(f"Adding memory {memory}")
        agent.add_memory(memory)

    return agent
