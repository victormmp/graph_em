import numpy as np
import pandas as pd


class GabrielGraph:

    def __init__(self, points: np.ndarray):
        self._points: np.ndarray = points
        self._connections: np.ndarray = np.zeros(points.shape)
        self._distances: np.ndarray = np.zeros(points.shape)

        self._do_calculations()

    def _do_calculations(self):
        self._calculate_distance()

    def _calculate_distance(self):
        for index_a in range(self._points.shape[0]):
            for index_b in range(index_a, self._points.shape[0]):
                continue

    @staticmethod
    def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        return np.linalg.norm(point_a - point_b)

    @staticmethod
    def _midpoint(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
        return (point_a + point_b) / 2

    @property
    def connections(self) -> np.ndarray:
        return self._connections

    def is_connection(self, point_a: int, point_b: int) -> bool:
        return (self._connections[point_a, point_b] == 1) or (self._connections[point_b, point_a] == 1)
