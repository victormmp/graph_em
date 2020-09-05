import numpy as np
import abc


class GeneticAlgorithm:

    def __init__(
            self,
            p_mutation: float = 0.05,
            p_cross: float = 0.3,
            pop_size: int = 30
    ):
        """
        Initialize the genetic algorithm with run parameters.

        Parameters
        ----------
        p_mutation: float
            Probability to mutate each gene of each sample.
        p_cross: float
            Probability to preform crossover between two samples in child generation.
        pop_size:
            Size of the population.
        """
        self.p_mutation = p_mutation
        self.p_cross = p_cross
        self.pop_size = pop_size

    def populate(self):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def cross_over(self):
        raise NotImplementedError

    
