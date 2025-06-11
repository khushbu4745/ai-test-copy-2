import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import uuid
import logging
from typing import Optional

class MemoryManager:
    def __init__(self, chroma_path="./chroma_db",
                 long_term_collection_name="creations_long_term",
                 short_term_collection_name="creations_short_term",
                 ollama_model_name="llama3", ollama_base_url="http://localhost:11434",
                 logger_instance: logging.Logger = None):
        """
        Initializes the MemoryManager with ChromaDB for long-term memory
        and an in-memory ChromaDB for short-term session context.
        Uses an Ollama model for embeddings.
        
        Args:
            chroma_path (str): Path for persistent ChromaDB.
            long_term_collection_name (str): Name for long-term collection.
            short_term_collection_name (str): Name for short-term collection.
            ollama_model_name (str): Ollama model name for embeddings.
            ollama_base_url (str): Base URL for the Ollama server.
            logger_instance (logging.Logger, optional): A logger instance to use for logging.
                                                      If None, a default logger will be created.
        """
        # Assign the provided logger instance, or get a default one
        self.logger = logger_instance if logger_instance is not None else logging.getLogger(__name__)
        if logger_instance is None:
            # If creating a default logger, ensure it has at least a console handler for visibility
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.setLevel(logging.INFO) # Set default level if no handlers
            self.logger.warning("No logger_instance provided to MemoryManager. Using default logger.")

        self.ollama_model_name = ollama_model_name
        self.ollama_base_url = ollama_base_url
        
        # Initialize the Ollama embedding function (shared for both long/short term)
        try:
            self.embedding_function = embedding_functions.OllamaEmbeddingFunction(
                url=self.ollama_base_url,
                model_name=self.ollama_model_name
            )
            self.logger.info(f"Ollama embedding function initialized with model '{self.ollama_model_name}'.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama embedding function: {e}")
            raise ConnectionError(f"Could not connect to Ollama at {self.ollama_base_url}. Please ensure Ollama is running and '{self.ollama_model_name}' model is available.") from e
        
        # --- Long-Term Memory ---
        self.chroma_path = chroma_path
        try:
            self.long_term_client = chromadb.PersistentClient(path=self.chroma_path)
            self.long_term_collection = self.long_term_client.get_or_create_collection(
                name=long_term_collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(f"Long-term ChromaDB collection '{long_term_collection_name}' initialized at: {self.chroma_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Long-term ChromaDB client or collection: {e}")
            raise

        # --- Short-Term Memory ---
        # Data is lost when the app restarts.
        try:
            self.short_term_client = chromadb.Client() # In-memory client
            self.short_term_collection = self.short_term_client.get_or_create_collection(
                name=short_term_collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(f"Short-term ChromaDB collection '{short_term_collection_name}' initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Short-term ChromaDB client or collection: {e}")
            raise


    def save_creation(self, user_prompt: str, expanded_prompt: str) -> Optional[str]:
        """
        Saves a new creative generation to both long-term
        and short-term memory.
        Returns a unique ID for the saved creation, or None if saving fails.
        """
        creation_id = str(uuid.uuid4()) # Use UUID for strong uniqueness

        metadata = {
            "user_prompt": user_prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        # --- Save to Long-Term Memory (Persistent ChromaDB) ---
        try:
            self.long_term_collection.add(
                documents=[expanded_prompt], # Embed the expanded prompt for semantic search
                metadatas=[metadata],
                ids=[creation_id]
            )
            self.logger.info(f"Creation '{creation_id}' saved to Long-term ChromaDB.")
        except Exception as e:
            self.logger.error(f"Error saving creation '{creation_id}' to Long-term ChromaDB: {e}")
            return None

        # --- Save to Short-Term Memory (In-Memory ChromaDB) ---
        try:
            self.short_term_collection.add(
                documents=[expanded_prompt], # Also embed the expanded prompt for short-term semantic search
                metadatas=[metadata],
                ids=[creation_id]
            )
            self.logger.info(f"Creation '{creation_id}' also added to Short-term ChromaDB.")
        except Exception as e:
            self.logger.error(f"Error saving creation '{creation_id}' to Short-term ChromaDB: {e}")
        
        return creation_id

    def _format_chroma_results(self, results) -> list[dict]:
        """Helper to format ChromaDB query results into a consistent dictionary list."""
        found_creations = []
        if results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                creation_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i] # This is the expanded_prompt that was embedded
                distance = results['distances'][0][i]

                found_creations.append({
                    "id": creation_id,
                    "user_prompt": metadata.get("user_prompt"),
                    "expanded_prompt": document,
                    "timestamp": metadata.get("timestamp"),
                    "similarity_distance": distance
                })
        return found_creations

    def search_short_term_memory(self, query: str, n_results: int = 1) -> list[dict]:
        """
        Searches short-term (in-memory ChromaDB) for semantically similar creations.
        """
        try:
            self.logger.info(f"Searching short-term memory for query: '{query[:50]}...' (n_results={n_results})")
            results = self.short_term_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            formatted_results = self._format_chroma_results(results)
            self.logger.info(f"Found {len(formatted_results)} results in short-term memory.")
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error searching short-term ChromaDB: {e}")
            return []

    def search_long_term_memory(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Searches long-term (persistent ChromaDB) for semantically similar creations.
        """
        try:
            self.logger.info(f"Searching long-term memory for query: '{query[:50]}...' (n_results={n_results})")
            results = self.long_term_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            formatted_results = self._format_chroma_results(results)
            self.logger.info(f"Found {len(formatted_results)} results in long-term memory.")
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error searching long-term ChromaDB: {e}")
            return []
        
    def select_best_memory_match(self, short_term_results: list[dict], long_term_results: list[dict]) -> tuple[Optional[dict], Optional[float]]:
        """
        Compares semantic matches from short-term and long-term memory, and selects the best match
        based on smallest similarity distance (higher semantic similarity).
        
        Returns:
            A tuple of (best_match_dict, best_score) or (None, None) if no match found.
        """
        self.logger.info("Evaluating best memory match from short-term and long-term results.")

        best_match = None
        best_score = None

        def compute_score(result: dict) -> Optional[float]:
            distance = result.get("similarity_distance")
            if distance is not None:
                score = distance
                self.logger.info(f"Result ID: {result['id']} | Distance: {distance:.4f} | Score: {score:.4f}")
                return score
            return None

        for source, results in [("short-term", short_term_results), ("long-term", long_term_results)]:
            for result in results:
                self.logger.info(f"result is {result}")
                score = compute_score(result)
                if score is not None and (best_score is None or score < best_score):
                    best_match = result
                    best_score = score
                    self.logger.info(f"New best match from {source} memory: result={result['user_prompt']} | Score={score:.4f}")

        if best_match:
            self.logger.info(f"Best match selected: ID={best_match['id']} with Score={best_score:.4f}")
        else:
            self.logger.info("No suitable memory match found.")

        return best_match, best_score

