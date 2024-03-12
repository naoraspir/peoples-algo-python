import logging
from multiprocessing import Pool
import os
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors

class FaceUniter:
    def __init__(self, cluster_reps, n_neighbors=5):
        self.cluster_reps = cluster_reps.copy()
        self.n_neighbors = n_neighbors
        self.num_processes = min(os.cpu_count() // 2, 4)
        logging.info(f"Number of processes: {self.num_processes}")
        logging.basicConfig(level=logging.DEBUG)
            
        self.embeddings = {}
        # Prepare embeddings for NearestNeighbors
        for rep in self.cluster_reps.keys():
            if cluster_reps[rep]['rep_embbeding'] is not None:
                self.embeddings[rep] = cluster_reps[rep]['rep_embbeding']
            else:
                logging.warning(f"Invalid embedding for rep_id {rep['rep_id']}. Skipping...")
        if len(self.embeddings) > 0:
            # Reshaping embeddings for NearestNeighbors
            self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric='cosine').fit(list(self.embeddings.values()))
            #log creation of nbrs
            logging.info("nbrs created")
        else:
            self.nbrs = None
            logging.warning("No valid embeddings found. Nearest Neighbors search will be skipped.")

    def update_look_alikes(self, rep_id):
        try:
            if not self.nbrs:
                logging.warning(f"No valid embeddings. Skipping update for rep_id {rep_id}.")
                return

            rep_embedding = [self.cluster_reps[rep_id]['rep_embbeding']]

            distances, indices = self.nbrs.kneighbors(rep_embedding)
            #log the indices for rep_id
            logging.info("rep_id:"+ str(rep_id)+ " indices: "+str(indices))

            # Exclude the first neighbor (the image itself)
            look_alikes_ids = [list(self.embeddings.keys())[i] for i in indices[0][1:]]
            self.cluster_reps[rep_id]['look_alikes'] = look_alikes_ids

            # Calculate and log average distance from look-alikes
            avg_distance = np.mean(distances[0][1:])
            logging.info(f"Avg distance from look-alikes for rep_id {rep_id}: {avg_distance}")

        except Exception as e:
            logging.error(f"Error updating look-alikes for rep_id {rep_id}: {e}")

    def run(self):
        if not self.nbrs:
            logging.warning("Face comparison will be skipped as there are no valid embeddings.")
            return self.cluster_reps

        logging.info("Starting face comparison...")
        # cluster_ids = list(self.cluster_reps.keys())

        # with Pool(processes=self.num_processes) as pool:
        #     pool.map(self.update_look_alikes, cluster_ids)
        for cluster_id in list(self.cluster_reps.keys()):
            logging.info("started look alike  process for cluster_id: "+str(cluster_id))
            self.cluster_reps[cluster_id]['look_alikes'] = []
            self.update_look_alikes(cluster_id)
            

        logging.info("Face comparison completed.")
        #log the cluster_reps amount 
        logging.info("cluster_reps len: "+str(len(self.cluster_reps)))
        return self.cluster_reps