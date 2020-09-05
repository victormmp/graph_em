from graph_em.graph.sampled import SampledGraph
import numpy as np
from typing import Union, List


class GraphSampler:
    """
    This class uses graph and an optimization routine to sample an optimal set of observations based on the maximization
    of the distances between them, considering all the observations of a given distribution.
    """

    def __init__(
            self,
            points: np.ndarray,
            fraction: Union[float, None] = 0.1,
            number_of_samples: Union[int, None] = None,
            random_generator: np.random.RandomState = np.random.RandomState()
    ):
        self.graph = SampledGraph(points, fraction, number_of_samples, random_generator)
        self.random = random_generator

    def sample(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError("The random sampler is not implemented yet.")
