import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self):
        self.client = None
        self.model = None
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "streamlit_documents")
        self.host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY", "")
        
    @st.cache_resource
    def get_qdrant_client(_self):
        """Initialize and return Qdrant client with caching"""
        try:
            if _self.api_key:
                client = QdrantClient(url=_self.host, api_key=_self.api_key)
            else:
                client = QdrantClient(url=_self.host)
            
            # Test connection
            client.get_collections()
            logger.info(f"Connected to Qdrant at {_self.host}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            st.error(f"Failed to connect to Qdrant: {e}")
            return None
    
    @st.cache_resource
    def get_embedding_model(_self):
        """Initialize and return sentence transformer model with caching"""
        try:
            # Using a lightweight model for better performance
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            st.error(f"Failed to load embedding model: {e}")
            return None
    
    def initialize(self):
        """Initialize Qdrant client and embedding model"""
        self.client = self.get_qdrant_client()
        self.model = self.get_embedding_model()
        
        if self.client and self.model:
            self._ensure_collection_exists()
            return True
        return False
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size
                vector_size = self.model.get_sentence_embedding_dimension()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            st.error(f"Error ensuring collection exists: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for given text"""
        if not self.model:
            logger.error("Embedding model not initialized")
            return None
        
        try:
            # Clean and prepare text
            cleaned_text = text.strip()
            if not cleaned_text:
                return None
            
            # Generate embedding
            embedding = self.model.encode(cleaned_text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """Add a document to the vector database"""
        if not self.client or not self.model:
            logger.error("Qdrant client or model not initialized")
            return False
        
        try:
            # Generate embedding
            embedding = self.get_embedding(content)
            if not embedding:
                return False
            
            # Prepare metadata
            payload = {
                "doc_id": doc_id,
                "content": content,
                "content_length": len(content),
                "timestamp": str(Path().cwd().stat().st_mtime)
            }
            if metadata:
                payload.update(metadata)
            
            # Create point
            point = PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload
            )
            
            # Add to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Added document {doc_id} to collection")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False
    
    def search_documents(self, query: str, top_k: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        """Search for similar documents using hybrid approach"""
        if not self.client or not self.model:
            logger.error("Qdrant client or model not initialized")
            return []

        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []

            # First: Semantic search with very low threshold to get more candidates
            semantic_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k * 3,  # Get more results for filtering
                score_threshold=score_threshold
            )

            # If no semantic results, try with even lower threshold
            if not semantic_results:
                semantic_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k * 3,
                    score_threshold=0.1  # Very low threshold as fallback
                )

            # Second: Keyword-based filtering for better relevance
            enhanced_results = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for result in semantic_results:
                content = result.payload.get("content", "").lower()
                doc_id = result.payload.get("doc_id", "")
                
                # Calculate keyword relevance
                keyword_matches = sum(1 for word in query_words if word in content)
                keyword_score = keyword_matches / len(query_words) if query_words else 0
                
                # Boost keyword matches significantly for better recall
                if keyword_score > 0:
                    # If we have keyword matches, boost the score
                    combined_score = (result.score * 0.4) + (keyword_score * 0.6)
                else:
                    # If no keyword matches, rely more on semantic similarity
                    combined_score = result.score
                
                enhanced_results.append({
                    "doc_id": doc_id,
                    "content": result.payload.get("content", ""),
                    "semantic_score": result.score,
                    "keyword_score": keyword_score,
                    "combined_score": combined_score,
                    "metadata": {k: v for k, v in result.payload.items() 
                              if k not in ["doc_id", "content"]}
                })
            
            # Sort by combined score and return top results
            enhanced_results.sort(key=lambda x: x["combined_score"], reverse=True)
            final_results = enhanced_results[:top_k]
            
            # Convert back to expected format
            results = []
            for result in final_results:
                results.append({
                    "doc_id": result["doc_id"],
                    "content": result["content"],
                    "score": result["combined_score"],  # Use combined score for display
                    "metadata": result["metadata"]
                })

            logger.info(f"Found {len(results)} documents for query: {query[:50]}...")
            
            # Debug logging
            if results:
                logger.info(f"Top result: {results[0].get('doc_id')} with score {results[0].get('score'):.3f}")
            else:
                logger.info("No results found - semantic search may be too restrictive")
            
            return results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector database"""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id]
            )
            logger.info(f"Deleted document {doc_id} from collection")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        if not self.client:
            return {}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": getattr(collection_info, 'name', self.collection_name),
                "vector_size": getattr(collection_info.config.params.vectors, 'size', 'N/A'),
                "distance": getattr(collection_info.config.params.vectors, 'distance', 'N/A'),
                "points_count": getattr(collection_info, 'points_count', 0)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

# Global instance
qdrant_manager = QdrantManager()

# Convenience functions
def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for text using the global Qdrant manager"""
    return qdrant_manager.get_embedding(text)

def search_qdrant(query: str, top_k: int = 5) -> List[Dict]:
    """Search Qdrant for similar documents"""
    return qdrant_manager.search_documents(query, top_k)

def add_document_to_qdrant(doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """Add document to Qdrant"""
    return qdrant_manager.add_document(doc_id, content, metadata)
