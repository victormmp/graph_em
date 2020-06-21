import click
import numpy as np
from typing import List, Tuple, Any


class SimpleUndirectedGraph:
    """
    A simple undirected graph class for easy instantiation and implementation of graphs.

    Attributes
    __________
    _points: numpy.ndarray
        The nodes location of the graph.
    _distances: numpy.ndarray
        A square matrix with the distance between each node of the graph.
    """

    def __init__(self, points: np.ndarray):
        """
        Builder method of Gabriel Graph Class. You need to give the point list as a numpy array, of the shape
        (number_of_points, number_of_dimension).
        For example, if dealing with 10 points in a R^2 dimension, the <points> object will have (10, 2) shape.

        Parameters
        ----------
        points: numpy.ndarray
            The input points to feed the graph.
        """

        self._points: np.ndarray = points
        self._distances: np.ndarray = np.zeros((points.shape[0], points.shape[0]))

        self._do_calculations()

    def _do_calculations(self):
        """
        Initialize the pipeline of calculations of the graph.
        Returns
        -------
        None.
        """

        self._calculate_distances()

    def _calculate_distances(self):
        """
        Calculate the distances of all nodes of the graph, using <self._points> as node locations.
        Returns
        -------
        None.
        """
        total_size: int = self._points.shape[0] ** 2
        with click.progressbar(length=total_size, label='Building Graph') as bar:
            for index_b in range(self._points.shape[0]):
                for index_a in range(index_b, self._points.shape[0]):

                    if index_b == index_a:
                        bar.update(1)
                        continue

                    distance = self._distance(point_a=self._points[index_a], point_b=self._points[index_b])
                    self._distances[index_a, index_b] = distance
                    bar.update(1)

    @property
    def distances(self) -> np.ndarray:
        upper_distances = self._distances.T
        return self._distances + upper_distances

    @property
    def edges(self) -> List[Tuple[Any, Any]]:
        point_b, point_a = np.where(self._distances > 0)
        point_a, point_b = [self._points[i, :] for i in point_a], [self._points[i, :] for i in point_b]

        return list(zip(point_a, point_b))

    @staticmethod
    def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        (Static Method) Calculate the Euclidian distance between two points in an Euclidian space.
        Parameters
        ----------
        point_a: numpy.ndarray
        point_b: numpy.ndarray

        Returns
        -------
        float
            Euclidian distance between <point_a> and <point_b>.
        """

        return np.linalg.norm(point_a - point_b)
