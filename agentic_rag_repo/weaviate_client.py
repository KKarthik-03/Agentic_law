# weaviate_client.py ---> Weaviate connection and retriever implementation

import streamlit as st
import weaviate

from typing import List
from pydantic import Field, PrivateAttr
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings

from config import WEA_URL, WEA_KEY

@st.cache_resource
def init_weaviate():
    """Initialize Weaviate client with caching."""
    if not WEA_URL or not WEA_KEY:
        return None
    try:
        if hasattr(weaviate, "connect_to_weaviate_cloud"):
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEA_URL,
                auth_credentials=weaviate.auth.AuthApiKey(WEA_KEY)
            )
        else:
            client = weaviate.Client(
                url=WEA_URL,
                additional_headers={"X-API-KEY": WEA_KEY}
            )
        return client
    except Exception as e:
        st.error(f"❌ Weaviate connection failed: {e}")
        return None


class WeaviateHybridRetriever(BaseRetriever):
    """Custom hybrid retriever for Weaviate using both semantic and keyword search"""

    client: weaviate.WeaviateClient = Field(..., description="Weaviate client instance.")
    collection_name: str = Field(..., description="Weaviate collection name.")
    embedding_model_name: str = Field(..., description="HuggingFace embedding model.")
    alpha: float = Field(0.5, description="Hybrid search alpha (0=keyword, 1=vector).")
    k: int = Field(5, description="Number of documents to retrieve.")

    _embeddings: HuggingFaceEmbeddings = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        """Retrieve documents using hybrid search"""
        try:
            collection = self.client.collections.get(self.collection_name)
            query_vector = self._embeddings.embed_query(query)

            results = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=self.alpha,
                limit=self.k,
                return_properties=["text", "case_title", "court", "file_name", "chunk_index"],
                return_metadata=weaviate.classes.query.MetadataQuery(score=True),
            )

            documents = []
            for obj in results.objects:
                props = obj.properties
                score = obj.metadata.score if obj.metadata and obj.metadata.score is not None else 0.0
                documents.append(
                    Document(
                        page_content=props.get("text", ""),
                        metadata={
                            "case_title": props.get("case_title", ""),
                            "court": props.get("court", ""),
                            "file_name": props.get("file_name", ""),
                            "chunk_index": props.get("chunk_index", 0),
                            "score": score,
                        },
                    )
                )
            return documents

        except Exception as e:
            st.error(f"❌ Error in retrieval: {e}")
            return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)