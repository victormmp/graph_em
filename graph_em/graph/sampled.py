import numpy as np
from .base import SimpleUndirectedGraph


class SampledGraph(SimpleUndirectedGraph):

    def __init__(self, points: np.ndarray, fraction: int = 0.1):

        points = np.random.choice()
        super(SampledGraph, self).__init__(points=points)
