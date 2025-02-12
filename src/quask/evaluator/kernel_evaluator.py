from abc import ABC, abstractmethod

import numpy as np

from ..core import Kernel


class KernelEvaluator(ABC):
    """Adds an evaluate functionality to the core kernel."""
    def __init__(self):
        """Init function."""
        self.last_result = None

    @abstractmethod
    def evaluate(
        self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray
    ):
        r"""Evaluate the current kernel and return the corresponding cost.
        
        Lower cost values corresponds to better solutions
        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """
