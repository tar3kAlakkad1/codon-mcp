import os
import uuid
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials
from constants import MARKDOWN_PATH, logger
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from google.cloud.firestore_v1.vector import Vector
from firebase_admin import firestore as admin_firestore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import UnstructuredMarkdownLoader

load_dotenv()
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
firebase_admin.initialize_app(cred)


class Embeddings:
    def __init__(self):
        self.db = admin_firestore.client()
        self.collection = "codon-docs"

    @classmethod
    def ingest_md_files(self) -> list[Document]:
        """
        Ingests all the documentation files for Codon.

        The base directory is /Users/tarekalakkadp/Desktop/personal/codon/docs.

        There are about 11 directories. Skip the 'developers' directory (contribution info)
        and the 'img' directory (images/SVGs). Include .md files in subdirectories.
        """
        docs = []

        for root, dirs, files in os.walk(MARKDOWN_PATH):
            dirs[:] = [d for d in dirs if d not in {"developers", "img"}]
            for file in files:
                if not file.endswith(".md"):
                    continue
                file_path = os.path.join(root, file)
                logger.info(f"Ingesting {file_path}")

                loader = UnstructuredMarkdownLoader(file_path)
                doc = loader.load()
                docs.extend(doc)

        logger.info(f"Ingested {len(docs)} documents.")
        return docs


class Chunk(Embeddings):
    db = admin_firestore.client()
    collection = "codon-docs"
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    chunker = SemanticChunker(embeddings, number_of_chunks=50)
    firestore_ref = db.collection(collection)

    @classmethod
    def split_docs(cls, docs: list[Document]) -> list[Document]:
        """
        Splits the documents into chunks.

        We will use the chunk size of 1000 tokens.
        """
        original_count = len(docs)
        chunks = cls.chunker.split_documents(docs)
        logger.info(f"Split {original_count} documents into {len(chunks)} chunks.")
        return chunks

    @classmethod
    def embed_docs(cls, docs: list[Document]) -> list[dict]:
        """
        Embeds the documents.
        """
        docs_with_embeddings = []
        texts = [doc.page_content for doc in docs]
        embedded_vectors = cls.embeddings.embed_documents(texts)

        for doc, embedding in zip(docs, embedded_vectors):
            prepared_chunk = cls.to_dict(doc, embedding)
            docs_with_embeddings.append(prepared_chunk)

        return docs_with_embeddings

    @classmethod
    def persist_docs(cls, docs: list[Document]) -> None:
        """
        Persists the documents to a vector store.

        We will use Firestore as the vector store.
        """
        docs = cls.embed_docs(docs)
        for doc in docs:
            cls.firestore_ref.add(doc, document_id=doc["id"])

        logger.info(f"Persisted {len(docs)} documents to Firestore.")

    @staticmethod
    def to_dict(doc: Document, embedding: list[float]) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "source": doc.metadata.get("source", ""),
            "content": doc.page_content,
            "embedding_field": Vector(embedding),
        }


def main():
    embeddings = Embeddings()
    docs = embeddings.ingest_md_files()
    chunked_docs = Chunk.split_docs(docs)
    Chunk.persist_docs(chunked_docs)


if __name__ == "__main__":
    main()
