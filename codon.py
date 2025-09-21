import httpx
from typing import Any
from google.cloud import firestore
from mcp.server.fastmcp import FastMCP
from constants import COLLECTION, logger, MCP_NAME, USER_AGENT
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

mcp = FastMCP(MCP_NAME)

def format_response(results: list[dict]) -> str:
    """
    Formats the response given the list of responses.
    """
    return "\n\n".join([result["content"] for result in results])

@mcp.tool()
def get_similar_docs(query: str, k: int = 5) -> list[dict]:
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

    formatted_response = format_response(results)

    return formatted_response


if __name__ == "__main__":
    mcp.run(transport="stdio")
