from ..graph.base import SimpleUndirectedGraph
import unittest
import numpy as np
from typing import List, Tuple


class TestSimpleGraph(unittest.TestCase):

    def setUp(self) -> None:
        """
        Sample points:

        0 --- 1 --- 2
        |     |     |
        3 --- 4 --- 5
        |     |     |
        6 --- 7 --- 8

        """
        self.points = np.array(
            [
                [-1,    1],
                [ 0,    1],
                [ 1,    1],
                [-1,    0],
                [ 0,    0],
                [ 1,    0],
                [-1,   -1],
                [ 0,   -1],
                [ 1,   -1]
            ]
        )

    def test_distance(self):
        point_a: np.ndarray = np.array([0, 0, 0])
        point_b: np.ndarray = np.array([2, 2, 2])

        distance = SimpleUndirectedGraph._distance(point_a, point_b)
        self.assertAlmostEqual(3.46, distance, 2)

    def test_build_distance_matrix(self):
        points = np.array(
            [
                [0,    1],
                [1,    1],
                [1,    0],
                [0,    0]
            ]
        )

        graph = SimpleUndirectedGraph(points=points)
        distances = np.array(
            [
                [0.0, 1.0, np.sqrt(2), 1.0],
                [1.0, 0.0, 1.0, np.sqrt(2)],
                [np.sqrt(2), 1.0, 0.0, 1.0],
                [1.0, np.sqrt(2), 1.0, 0.0]
            ]
        )

        self.assertTrue(np.array_equal(distances, graph.distances))

    def tearDown(self) -> None:
        pass

