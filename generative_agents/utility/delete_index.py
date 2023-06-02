import pinecone
import os
import argparse

from generative_agents.utility import logger

log = logger.init("delete_index")

def delete_index(index_name):
    log.info(f"Deleting pinecone index {index_name}..")
    input()
    pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
        )
    pinecone.delete_index("generative-agents-index")
    log.info(f"Successfully deleted index {index_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete Pinecone Index Script")
    parser.add_argument(
        "--index",
        type=str,
        default="generative-agents-index",
        action="store",
        dest="index",
    )
    args = vars(parser.parse_args())

    delete_index(args["index"])