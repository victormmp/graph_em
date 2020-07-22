from graph_em.graph.sampled import SampledGraph
import numpy as np
from typing import Union, List


class GraphSampler:

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
