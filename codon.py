import httpx
from typing import Any
from dotenv import load_dotenv
from google.cloud import firestore
from mcp.server.fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from google.cloud.firestore_v1.vector import Vector
from constants import COLLECTION, logger, MCP_NAME, USER_AGENT
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

mcp = FastMCP(MCP_NAME)

@mcp.tool()
def get_similar_docs(query: str, k: int = 5) -> list[str]:
    """
    Gets the most similar documents to the query.

    Args:
        query (str): The user / agent query.
        k (int): The number of similar documents to return.

    Returns:
        list[dict]: The most similar documents to the query.
    """
    db = firestore.Client()
    ref = db.collection(COLLECTION)

    load_dotenv()
    embedded_query = OpenAIEmbeddings().embed_query(query)
    vector_query = ref.find_nearest(
        vector_field="embedding_field",
        query_vector=embedded_query,
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=k,
    )

    docs = vector_query.stream()
    results = [doc.to_dict()["content"] for doc in docs]

    return results


if __name__ == "__main__":
    mcp.run(transport="stdio")
