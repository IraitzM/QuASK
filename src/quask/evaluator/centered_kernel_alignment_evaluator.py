import numpy as np

from ..core import Kernel
from .kernel_alignment_evaluator import KernelAlignmentEvaluator
from .kernel_evaluator import KernelEvaluator


class CenteredKernelAlignmentEvaluator(KernelEvaluator):
    """Kernel compatibility measure based on the centered kernel-target alignment.

    See: "Algorithms for learning kernels based on centered alignment."
    Cortes, Corinna, Mehryar Mohri, and Afshin Rostamizadeh.
    The Journal of Machine Learning Research 13.1 (2012): 795-828.
    """

    def evaluate(
        self, kernel: Kernel, K: np.ndarray, X: np.ndarray, y: np.ndarray
    ):
        r"""Evaluate the current kernel and return the corresponding cost.
        
        Lower cost values corresponds to better solutions.

        :param kernel: kernel object
        :param K: optional kernel matrix \kappa(X, X)
        :param X: datapoints
        :param y: labels
        :return: cost of the kernel, the lower the better
        """
        if K is None:
            K = kernel.build_kernel(X, X)
        Kc = self.center_kernel(K)
        kta = KernelAlignmentEvaluator.kta(Kc, y)
        return -np.abs(kta)

    @staticmethod
    def center_kernel(K):
        """Center a kernel (subtract its mean value).

        :param K: kernel matrix
        :return: centered kernel
        """
        m = K.shape[0]
        U = np.eye(m) - (1 / m) * np.outer([1] * m, [1] * m)
        Kc = U @ K @ U.T
        return Kc
