from graph_em.graph.sampled import SampledGraph
import unittest
import numpy as np


class TestSampleGraph(unittest.TestCase):

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

        cls.rg = np.random.RandomState(seed=42)
        cls.rg_initial_state = cls.rg.get_state()

    def test_sampled_points(self):
        with self.subTest("Sampling using fixed number of samples"):
            self.rg.set_state(self.rg_initial_state)
            graph = SampledGraph(self.points, random_generator=self.rg, number_of_samples=4, fraction=None)
            sampled_points = np.array(
                [
                    [ 0, -1],
                    [ 0,  1],
                    [ 1,  0],
                    [-1,  1]
                ]
            )

            self.assertTrue(np.array_equal(sampled_points, graph._points))

        with self.subTest("Sampling using fraction"):
            self.rg.set_state(self.rg_initial_state)
            graph = SampledGraph(self.points, random_generator=self.rg, number_of_samples=None, fraction=0.3)
            sampled_points = np.array(
                [
                    [ 0, -1],
                    [ 0,  1]
                ]
            )

            self.assertTrue(np.array_equal(sampled_points, graph._points))

        with self.subTest("Using both number_of_samples and fraction"):
            self.rg.set_state(self.rg_initial_state)
            graph = SampledGraph(self.points, random_generator=self.rg, number_of_samples=4, fraction=0.3)
            sampled_points = np.array(
                [
                    [0, -1],
                    [0, 1],
                    [1, 0],
                    [-1, 1]
                ]
            )

            self.assertTrue(np.array_equal(sampled_points, graph._points))

        with self.subTest("Without passing any of number_of_samples and fraction."):
            with self.assertRaises(AttributeError):
                graph = SampledGraph(self.points, random_generator=self.rg, number_of_samples=None, fraction=None)

    def tearDown(self) -> None:
        pass

