"""
Source: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py

Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance are either raw features or if a model
has transform method then uses the output of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of outlier datapoints.
Resulting centers are solution to multiple integer program.
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from typing import List, Optional, Sequence, Union


class CoreSetSelector:
    def __init__(
            self,
            features: Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]],
            *,
            seed: Optional[Union[int, np.random.Generator]] = 42,
            metric: Optional[str] = 'euclidean'):
        self.seed = seed
        self.metric = metric
        self.min_distances = None
        self.already_selected = []
        self.features = features
        self.n_obs = self.features.shape[0]

    def _update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.

        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select(self, budget: int = 2) -> List[int]:
        """
        Diversity promoting active learning method that greedily forms a batch to minimize the maximum distance to a
        cluster center among all unlabeled datapoints.
        Returns:
            indices of points selected to minimize distance to cluster centers
        """
        self._update_distances([], only_new=False, reset_dist=True)

        for _ in range(budget):
            if len(self.already_selected) == 0:
                # Initialize centers with a randomly selected datapoint
                np.random.seed(self.seed)
                ind = np.random.choice(np.arange(self.n_obs))
                print(f"Randomly selected point: {ind} with seed {self.seed}")
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in self.already_selected

            self._update_distances([ind], only_new=True, reset_dist=False)
            self.already_selected.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        return self.already_selected
