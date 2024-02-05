import json
import os
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf.struct_pb2 import Value

class ClusterSearchIndexer:
    def __init__(self, session_key, bucket_name):
        # Automatically determine the project ID and location from the environment
        self.project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.location = os.environ.get('CLOUD_RUN_REGION', 'us-east1')  # Default to 'us-central1' if not set
        self.session_key = session_key
        self.bucket_name = bucket_name
        self.index_display_name = session_key
        self.aiplatform_client = aiplatform.init(project=self.project_id, location=self.location)
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        self.clusters_folder = f"{session_key}/clusters/"

    def load_centroids_from_gcs(self):
        centroids_blob = self.bucket.blob(f"{self.clusters_folder}cluster_centroids.json")
        centroids_data = centroids_blob.download_as_string()
        return json.loads(centroids_data)

    def create_matching_engine_index(self):
        centroids_URI = f"{self.clusters_folder}cluster_centroids.json"
        # create Index
        my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name = self.index_display_name,
            contents_delta_uri = centroids_URI,
            dimensions = 512,
            approximate_neighbors_count = 4,
        )

        ## create `IndexEndpoint`
        my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name = self.index_display_name,
            public_endpoint_enabled = True
        )

        # deploy the Index to the Index Endpoint
        my_index_endpoint.deploy_index(
            index = my_index, deployed_index_id = self.index_display_name
        )

    def upload_centroids_to_vector_search(self):
        # centroids = self.load_centroids_from_gcs()

        # Create or get your Matching Engine index
        index = self.create_matching_engine_index()

        # # Prepare your centroids for upload
        # for cluster_id, centroid in centroids.items():
        #     # Convert your centroid data to the format expected by Vertex AI
        #     vector = Value(list_value=Value.ListValue(values=[Value(number_value=v) for v in centroid]))

        #     # Upload each centroid to the index
        #     # Note: You might need to batch this if you have many centroids
        #     self.aiplatform_client.upsert_data_items(
        #         index=index.name,
        #         requests=[{
        #             "data_item_id": cluster_id,
        #             "value": vector
        #         }]
        #     )

    # Add any additional methods for vector search operations as needed
