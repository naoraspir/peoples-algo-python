import json
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from pinecone.core.openapi.shared.exceptions import PineconeException
import logging
import numpy as np

from common.consts_and_utils import (
    PINCONE_API_KEY,
    PINCONE_ENNVIROMENT,
    PINECONE_DEFAULT_EMBBEDING_DIM,
    PINECONE_DISTANCE_METRIC,
    PINECONE_INDEX_NAME,
    PINECONE_UPSERT_BATCH_SIZE
)

class FaceIndexer:
    def __init__(self, session_key: str, dimension: int = PINECONE_DEFAULT_EMBBEDING_DIM):
        """
        Initializes the FaceIndexer with a given session_key (namespace) and a constant index.
        Deletes the namespace if it exists and recreates it.
        
        Args:
            session_key: The session_key used as the namespace.
            dimension: The dimensionality of the vectors (default is 512).
        """
        self.index_name = PINECONE_INDEX_NAME  # Constant index
        self.dimension = dimension
        self.namespace = session_key  # Use session_key as the namespace

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=PINCONE_API_KEY)

        # Check if the index exists using list_indexes()
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            logging.info(f"Index {self.index_name} does not exist. Creating it.")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=PINECONE_DISTANCE_METRIC,
                spec=ServerlessSpec(
                    cloud="gcp",
                    region=PINCONE_ENNVIROMENT
                )
            )

        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        logging.info(f"Successfully connected to index {self.index_name}.")

        # Delete the namespace if it exists
        try:
            logging.info(f"Deleting existing namespace: {self.namespace}")
            self.index.delete(delete_all=True, namespace=self.namespace)
        except PineconeException as e:
            if "Namespace not found" in str(e):
                logging.info(f"Namespace {self.namespace} not found. Skipping deletion.")
            else:
                raise  # Re-raise the exception if it's another issue

    def chunker(self, seq, batch_size):
        """
        Splits the list into batches of the specified size.
        
        Args:
            seq: The sequence to split into chunks.
            batch_size: The size of each chunk.
            
        Returns:
            A generator of chunked data.
        """
        return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

    def index_faces(self, embeddings: list, image_paths: list, metrics:list, batch_size: int = PINECONE_UPSERT_BATCH_SIZE):
        """
        Indexes a list of face embeddings (NumPy arrays) with corresponding image paths into the namespace in batches.
        The upsert operation is performed asynchronously.

        Args:
            embeddings: A list of face embeddings (NumPy arrays) to index.
            image_paths: A list of corresponding image paths for each embedding.
            batch_size: The size of each batch (default is 200).
        """
        logging.info(f"Upserting {len(embeddings)} face embeddings into namespace {self.namespace} in batches of {batch_size}.")

        if len(embeddings) != len(image_paths):
            raise ValueError("The number of embeddings and image paths must be the same.")

        # Prepare the vectors with metadata
        vectors_to_index = []
        for i, embedding in enumerate(embeddings):
            # Ensure that embeddings are converted from NumPy arrays to Python lists
            embedding_values = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            # Add image path and metrics to the metadata
            metadata = {
                'image_path': image_paths[i],
                'metrics': json.dumps(metrics[i])  # Add metrics as part of the metadata
            }
            
            # Each vector will be indexed by a unique id, such as 'session_key_i'
            vectors_to_index.append({
                "id": f'{self.namespace}_{i}',
                "values": embedding_values,
                "metadata": metadata
            })

        # Perform asynchronous upserts in batches
        async_results = []
        for chunk in self.chunker(vectors_to_index, batch_size):
            async_results.append(self.index.upsert(vectors=chunk, namespace=self.namespace, async_req=True))

        # Wait for and retrieve responses (in case of error)
        [async_result.result() for async_result in async_results]

        logging.info(f"Successfully indexed {len(vectors_to_index)} face embeddings into namespace {self.namespace} in batches.")

    def check_index_ready(self):
        """
        Check if the index is ready by polling the status of the index.

        Returns:
            True if the index is ready, False otherwise.
        """
        index_status = self.pc.describe_index(self.index_name).status
        if index_status['state'] == 'Ready':
            logging.info(f"Index {self.index_name} is ready.")
            return True
        else:
            logging.info(f"Index {self.index_name} is not ready. Current state: {index_status['state']}")
            return False

    def delete_namespace(self):
        """
        Deletes the current namespace from the Pinecone index.
        """
        logging.info(f"Deleting namespace {self.namespace}.")
        self.index.delete(delete_all=True, namespace=self.namespace)
