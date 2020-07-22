import unittest
import numpy as np
from graph_em.sampler.graph_sampler import GraphSampler


class TestGraphSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rg = np.random.RandomState(seed=42)

        clusters =

        cls.rg_initial_state = cls.rg.get_state()



    def setUp(self) -> None:
        self.rg.set_state(self.rg_initial_state)

    def test_generate_sample(self):
        sampler = GraphSampler()

    def tearDown(self) -> None:
        pass


if __name__ == '__main__':
    unittest.main()
