import math

import faiss
import pinecone
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever, PineconeHybridSearchRetriever
from langchain.vectorstores import FAISS, Pinecone

from pinecone_text.sparse import BM25Encoder


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever(is_pinecone: bool) -> None:
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    embedding_dim = 1536
    # Initialize the vectorstore as empty
    if is_pinecone:
        index = pinecone.Index("generative-agents-index")
        # use default tf-idf values
        bm25_encoder = BM25Encoder().default()
        return PineconeHybridSearchRetriever(
            embeddings=embeddings_model,
            sparse_encoder=bm25_encoder,
            index=index,
        )
    else:
        index = faiss.IndexFlatL2(embedding_dim)
        vectorstore = FAISS(
            embeddings_model.embed_query,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=relevance_score_fn,
        )
        return TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore, other_score_keys=["importance"], k=15
        )

