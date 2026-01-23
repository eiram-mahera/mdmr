"""
This code introduces a new class KMeansBasedSampling. It takes all_styles (your feature vectors) and n_samples_to_pick (defaulting to 5) during initialization. The find_query_sample method performs the following steps:

It initializes and fits a KMeans model to your all_styles data, creating n_samples_to_pick clusters.
It then retrieves the centroids of these clusters.
For each centroid, it calculates the Euclidean distance to all samples in all_styles and selects the index of the sample that is closest to that centroid.
Finally, it returns a list of these selected indices, ensuring that only unique indices are returned in case multiple centroids happen to be closest to the same sample.
"""
from typing import List, Sequence, Union
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class KMeansClusteringSelector:
    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
    ):
        self.features = features

    def select(self, budget: int = 2) -> List[int]:
        # Initialize and fit KMeans
        # n_init='auto' is recommended for robust centroid initialization
        kmeans = KMeans(
            n_clusters=budget,
            init='k-means++',
            random_state=budget,
            n_init='auto'
        )
        kmeans.fit(self.features)

        # Get the cluster centroids
        centroids = kmeans.cluster_centers_

        selected_indices = []
        used_indices = set()

        # For each centroid, find the sample in all_styles that is closest to it
        for i in range(budget):
            centroid = centroids[i]

            # Calculate distances from this centroid to all samples
            distances_to_centroid = pairwise_distances(self.features, centroid.reshape(1, -1)).flatten()

            # Sort the indices by distance to centroid
            sorted_indices = np.argsort(distances_to_centroid)

            # Select the closest index that has not been used yet
            for idx in sorted_indices:
                if idx not in used_indices:
                    selected_indices.append(idx)
                    used_indices.add(idx)
                    break

        return selected_indices

