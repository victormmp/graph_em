import numpy as np
import pandas as pd
import click


class GabrielGraph:

    def __init__(self, points: np.ndarray):
        self._points: np.ndarray = points
        self._connections: np.ndarray = np.zeros((points.shape[0], points.shape[0]))
        self._distances: np.ndarray = np.zeros((points.shape[0], points.shape[0]))

        self._do_calculations()

    def _do_calculations(self):
        self._calculate_distance()

    def _calculate_distance(self):
        self._connections = np.tril(np.ones(self._connections.shape), k=-1)
        total_size = self._connections.shape[0] ** 3
        with click.progressbar(length=total_size, label='Building Graph') as bar:
            for index_b in range(self._points.shape[0]):
                for index_a in range(index_b, self._points.shape[0]):
                    midpoint = self._midpoint(point_a=self._points[index_a], point_b=self._points[index_b])
                    distance = self._distance(point_a=self._points[index_a], point_b=self._points[index_b])
                    self._distances[index_a, index_b] = distance
                    distance /= 2
                    for index_other_point in range(self._points.shape[0]):
                        bar.update(1)
                        if (index_other_point == index_a) or (index_other_point == index_b):
                            continue
                        distance_other_point = self._distance(point_a=midpoint, point_b=self._points[index_other_point])
                        if distance_other_point < distance:
                            self._connections[index_a, index_b] = 0
                            break

    @staticmethod
    def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        return np.linalg.norm(point_a - point_b)

    @staticmethod
    def _midpoint(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
        return (point_a + point_b) / 2

    @property
    def connections(self) -> np.ndarray:
        upper_connections = self._connections.T
        return self._connections + upper_connections

    def is_connection(self, point_a: int, point_b: int) -> bool:
        return (self._connections[point_a, point_b] == 1) or (self._connections[point_b, point_a] == 1)
