import numpy as np
from .base import SimpleUndirectedGraph


class SampledGraph(SimpleUndirectedGraph):

    def __init__(self, points: np.ndarray):


        super(SampledGraph, self).__init__(points=points)
