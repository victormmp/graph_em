import click
import numpy as np


class GabrielGraph:
    """
    Gabriel graph class for easy instantiation and implementation of Gabriel Graphs.

    Attributes
    __________
    _points: numpy.ndarray
        The nodes location of the Gabriel Graph.
    _connections: numpy.ndarray
        A binary triangular matrix indicating if (line) have a connection with (column) according to Gabriel Graph
        definition.
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
        self._connections: np.ndarray = np.zeros((points.shape[0], points.shape[0]))
        self._distances: np.ndarray = np.zeros((points.shape[0], points.shape[0]))

        self._do_calculations()

    def _do_calculations(self):
        """
        Initialize the pipeline of calculations of the graph.
        Returns
        -------
        None.
        """

        self._calculate_connections()

    def _calculate_connections(self):
        """
        Calculate with nodes are connected according to Gabriel Graph definition, using <self._points> as node
        locations, registering which nodes are connected.
        Returns
        -------
        None.
        """
        self._connections = np.tril(np.ones(self._connections.shape), k=-1)
        total_size = self._connections.shape[0] ** 3
        with click.progressbar(length=total_size, label='Building Graph') as bar:
            for index_b in range(self._points.shape[0]):
                for index_a in range(index_b, self._points.shape[0]):

                    if index_b == index_a:
                        bar.update(1)
                        continue

                    midpoint = self._midpoint(point_a=self._points[index_a], point_b=self._points[index_b])
                    distance = self._distance(point_a=self._points[index_a], point_b=self._points[index_b])
                    self._distances[index_a, index_b] = distance
                    distance /= 2

                    number_of_points_analysed = 0

                    for index_other_point in range(self._points.shape[0]):

                        if (index_other_point == index_a) or (index_other_point == index_b):
                            bar.update(1)
                            continue

                        distance_other_point = self._distance(point_a=midpoint, point_b=self._points[index_other_point])

                        if distance_other_point < distance:
                            self._connections[index_a, index_b] = 0
                            bar.update(self._points.shape[0] - number_of_points_analysed)
                            break

                        number_of_points_analysed += 1
                        bar.update(1)

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

    @staticmethod
    def _midpoint(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
        """
        (Static Method) Calculate the intermediate point between two initial points, with the same distance between
        both.
        Parameters
        ----------
        point_a: numpy.ndarray
        point_b: numpy.ndarray

        Returns
        -------
        numpy.ndarray
            The intermediate point location as a numpy array, with the shape of the space dimension.
        """
        return (point_a + point_b) / 2

    @property
    def connections(self) -> np.ndarray:
        """
        Python parameter. Return the connection matrix as a squared symmetric binary matrix.
        Returns
        -------
        numpy.ndarray
            Squared symmetric binary matrix.
        """
        upper_connections = self._connections.T
        return self._connections + upper_connections

    def is_connection(self, point_a: int, point_b: int) -> bool:
        """
        Indicate if <point_a> and <point_b> are connected in a Gabriel Graph.
        Parameters
        ----------
        point_a: int
        point_b: int

        Returns
        -------
        bool
            If the two input points are connected or not.
        """
        return (self._connections[point_a, point_b] == 1) or (self._connections[point_b, point_a] == 1)
