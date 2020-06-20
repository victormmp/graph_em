from ..graph.gabriel_graph import GabrielGraph
import unittest
import numpy as np
from typing import List, Tuple


class TestGabrielGraph(unittest.TestCase):

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

    def test_connections(self):
        expected_connections: np.ndarray = np.zeros((self.points.shape[0], self.points.shape[0]))
        connected_points: List[List] = [
            [1, 0],
            [2, 1],
            [4, 3],
            [5, 4],
            [7, 6],
            [8, 7],
            [3, 0],
            [6, 3],
            [4, 1],
            [7, 4],
            [5, 2],
            [8, 5],
            [4, 0],
            [8, 4],
            [4, 2],
            [6, 4],
            [7, 5],
            [3, 1],
            [5, 1],
            [7, 3]
        ]

        for point_a, point_b in connected_points:
            expected_connections[point_a, point_b] = 1

        graph = GabrielGraph(points=self.points)

        with self.subTest('Assert triangular matrix of connections private property'):
            connections = graph._connections

            self.assertTrue(
                np.array_equal(
                    np.zeros((self.points.shape[0], self.points.shape[0])),
                    np.triu(connections)
                )
            )

        with self.subTest('Assert symmetric matrix of connections public property'):
            expected_symmetric_connections: np.ndarray = np.zeros((self.points.shape[0], self.points.shape[0]))
            for point_a, point_b in connected_points:
                expected_symmetric_connections[point_a, point_b] = 1
                expected_symmetric_connections[point_b, point_a] = 1

            connections = graph.connections

            self.assertTrue(
                np.array_equal(
                    expected_symmetric_connections,
                    connections
                )
            )

    def test_distance(self):
        point_a: np.ndarray = np.array([0, 0, 0])
        point_b: np.ndarray = np.array([2, 2, 2])

        distance = GabrielGraph._distance(point_a, point_b)
        self.assertAlmostEqual(3.46, distance, 2)

    def test_midpoint(self):
        points_list: List[List[List]] = [
            [[0, 0, 0], [2, 2, 2]],
            [[-1, -1, -1, -1], [1, 1, 1, 1]],
            [[-1, -1, -1, -1], [1, 1, 1, 0]]
        ]

        midpoints: List[np.ndarray] = [
            np.array([1, 1, 1]),
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, -0.5])
        ]

        for points, midpoint in zip(points_list, midpoints):
            with self.subTest(f"Testing point {points[0]} and {points[1]}"):
                point_a = np.array(points[0])
                point_b = np.array(points[1])

                calculated_midpoint = GabrielGraph._midpoint(point_a, point_b)
                self.assertTrue(np.array_equal(midpoint, calculated_midpoint))

    def tearDown(self) -> None:
        pass

