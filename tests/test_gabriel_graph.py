from graph_em.graph.gabriel_graph import GabrielGraph
import unittest
import numpy as np
from typing import List


class TestGabrielGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sample points:

        0 --- 1 --- 2
        |     |     |
        3 --- 4 --- 5
        |     |     |
        6 --- 7 --- 8

        """
        cls.points = np.array(
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

        cls.connected_points: List[List] = [
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

        cls.expected_connections: np.ndarray = np.zeros((cls.points.shape[0], cls.points.shape[0]))

        for point_a, point_b in cls.connected_points:
            cls.expected_connections[point_a, point_b] = 1

        cls.expected_symmetric_connections: np.ndarray = np.zeros((cls.points.shape[0], cls.points.shape[0]))
        for point_a, point_b in cls.connected_points:
            cls.expected_symmetric_connections[point_a, point_b] = 1
            cls.expected_symmetric_connections[point_b, point_a] = 1

    def test_connections(self):

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
            connections = graph.connections

            self.assertTrue(
                np.array_equal(
                    self.expected_symmetric_connections,
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

    def test_is_connection(self):
        graph = GabrielGraph(points=self.points)

        for point_a in range(self.points.shape[0]):
            for point_b in range(self.points.shape[0]):
                with self.subTest(f"Check connection between point {point_a} and {point_b}"):
                    self.assertEqual(
                        self.expected_symmetric_connections[point_a, point_b], graph.is_connection(point_a, point_b)
                    )

    def test_edges(self):

        expected_edges = []
        for point_b, point_a in self.connected_points:
            expected_edges.append(np.array([self.points[point_a], self.points[point_b]]))

        graph = GabrielGraph(self.points)

        for edge in graph.edges:
            e = np.array(edge)
            self.assertTrue(any([np.allclose(e, expec_edge) for expec_edge in expected_edges]))

    def tearDown(self) -> None:
        pass

