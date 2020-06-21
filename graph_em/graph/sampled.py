import numpy as np
from .base import SimpleUndirectedGraph
from typing import Union


class SampledGraph(SimpleUndirectedGraph):

    def __init__(
            self,
            points: np.ndarray,
            fraction: Union[float, None] = 0.1,
            number_of_samples: Union[int, None] = None,
            random_generator: np.random.RandomState = np.random.RandomState()
    ):
        if not number_of_samples:
            if not fraction:
                raise AttributeError("Number of samples (number_of_samples) or a fraction from the number of points "
                                 "(fraction) must be informed and valid. Please check those parameters.")

            number_of_samples = int(fraction * points.shape[0])

        self.random_generator = random_generator
        sampled_points = points[self.random_generator.choice(points.shape[0], number_of_samples, replace=False), :]

        super(SampledGraph, self).__init__(points=sampled_points)
