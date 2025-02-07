"""Basic unit testing for the PennyLane kernels."""
import unittest

import numpy as np
from quask.core import Ansatz, Kernel, KernelType
from quask.core_implementation import PennylaneKernel


class TestKernel(unittest.TestCase):
    """Testing basic functionalities on kernels."""

    def test_static_single_qubit(self):
        """Test static single qubit."""
        # circuit:  |0> - RX(pi) -
        #           |0> - ID     -
        ansatz = Ansatz(n_features=1, n_qubits=2, n_operations=1)
        ansatz.initialize_to_identity()
        ansatz.change_generators(0, "XI")
        ansatz.change_feature(0, -1)
        ansatz.change_wires(0, [0, 1])
        ansatz.change_bandwidth(0, 1)

        # measurement operation = <1|Z|1>
        # probabilities: [0.0, 1.0]
        # observable: 0.0 * (+1) + 1.0 * (-1) = -1.0
        kernel = PennylaneKernel(ansatz, "ZI", KernelType.OBSERVABLE)
        x = kernel.phi(np.array([np.inf]))
        assert np.allclose(kernel.get_last_probabilities(), np.array([0, 1])), (
            f"Incorrect measurement: {kernel.get_last_probabilities()}"
        )
        assert np.isclose(x, -1), "Incorrect observable"

        # measurement operation = <1|X|1> = <1H|Z|H1> = <+|Z|+>
        # probabilities: [0.5, 0.5]
        # observable: 0.5 * (+1) + 0.5 * (-1) = 0.0
        kernel = PennylaneKernel(ansatz, "XI", KernelType.OBSERVABLE)
        x = kernel.phi(np.array([np.inf]))
        assert np.allclose(
            kernel.get_last_probabilities(), np.array([0.5, 0.5])
        ), f"Incorrect measurement: {kernel.get_last_probabilities()}"
        print(x)
        assert np.isclose(x, 0), "Incorrect observable"

        # measurement operation
        # <1|Y|1> = <1HSdag|Z|SdagH1> =
        # = <[1/sqrt(2), -i/sqrt(2)]|Z|[1/sqrt(2), -i/sqrt(2)]>
        # probabilities: [0.5, 0.5]
        # observable: 0.5 * (+1) + 0.5 * (-1) = 0.0
        kernel = PennylaneKernel(ansatz, "YI", KernelType.OBSERVABLE)
        x = kernel.phi(np.array([np.inf]))
        assert np.allclose(
            kernel.get_last_probabilities(), np.array([0.5, 0.5])
        ), f"Incorrect measurement: {kernel.get_last_probabilities()}"
        assert np.isclose(x, 0), "Incorrect observable"


    def test_static_two_qubit(self):
        """Two qubit case."""
        # circuit:  |0> - XY(#0) -
        #           |0> - XY(#0) -
        ansatz = Ansatz(n_features=1, n_qubits=2, n_operations=1)
        ansatz.initialize_to_identity()
        ansatz.change_generators(0, "XY")
        ansatz.change_feature(0, 0)
        ansatz.change_wires(0, [0, 1])
        ansatz.change_bandwidth(0, 1)

        kernel = PennylaneKernel(ansatz, "ZZ", KernelType.OBSERVABLE)
        x = kernel.phi(np.array([np.pi / 2]))
        assert np.allclose(
            kernel.get_last_probabilities(), np.array([0.5, 0.0, 0.0, 0.5])
        ), "Incorrect measurement"
        assert np.isclose(x, 0), "Incorrect observable"


    def _check_kernel_value(self, kernel: Kernel, x1: float, x2: float, expected: float):
        similarity = kernel.kappa(x1, x2)
        print(similarity, expected)
        assert np.isclose(similarity, expected), (
            f"Kernel value is {similarity:0.3f} while {expected:0.3f} was expected"
        )


    def check_kernel_rx_value(kernel: Kernel, x1: float, x2: float):
        def rx(theta):
            return np.array(
                [
                    [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                    [-1j * np.sin(theta / 2), np.cos(theta / 2)],
                ]
            )

        ket_zero = np.array([[1], [0]])
        ket_phi = np.linalg.inv(rx(x2)) @ rx(x1) @ ket_zero
        expected_similarity = (np.abs(ket_phi[0]) ** 2).real
        check_kernel_value(
            kernel, np.array([x1]), np.array([x2]), expected_similarity
        )

    def test_rx_kernel_fidelity():
        ansatz = Ansatz(
            n_features=1,
            n_qubits=2,
            n_operations=1,
            allow_midcircuit_measurement=False,
        )
        ansatz.initialize_to_identity()
        ansatz.change_operation(
            0,
            new_feature=0,
            new_wires=[0, 1],
            new_generator="XI",
            new_bandwidth=1.0,
        )
        kernel = PennylaneKernel(
            ansatz,
            "ZZ",
            KernelType.FIDELITY,
            device_name="default.qubit",
            n_shots=None,
        )

        check_kernel_value(kernel, np.array([0.33]), np.array([0.33]), 1.0)

        check_kernel_rx_value(kernel, 0.00, 0.00)
        check_kernel_rx_value(kernel, 0.33, 0.33)
        check_kernel_rx_value(kernel, np.pi / 2, np.pi / 2)
        check_kernel_rx_value(kernel, np.pi, np.pi)
        check_kernel_rx_value(kernel, 0, np.pi)
        check_kernel_rx_value(kernel, 0.33, np.pi)
        check_kernel_rx_value(kernel, np.pi / 2, np.pi)
        check_kernel_rx_value(kernel, 0, 0.55)
        check_kernel_rx_value(kernel, 0.33, 0.55)
        check_kernel_rx_value(kernel, np.pi / 2, 0.55)


    def test_rx_kernel_fidelity():
        ansatz = Ansatz(
            n_features=1,
            n_qubits=2,
            n_operations=1,
            allow_midcircuit_measurement=False,
        )
        ansatz.initialize_to_identity()
        ansatz.change_operation(
            0,
            new_feature=0,
            new_wires=[0, 1],
            new_generator="XI",
            new_bandwidth=1.0,
        )
        kernel = PennylaneKernel(
            ansatz,
            "ZZ",
            KernelType.SWAP_TEST,
            device_name="default.qubit",
            n_shots=None,
        )

        check_kernel_value(kernel, np.array([0.33]), np.array([0.33]), 1.0)

        check_kernel_rx_value(kernel, 0.00, 0.00)
        check_kernel_rx_value(kernel, 0.33, 0.33)
        check_kernel_rx_value(kernel, np.pi / 2, np.pi / 2)
        check_kernel_rx_value(kernel, np.pi, np.pi)
        check_kernel_rx_value(kernel, 0, np.pi)
        check_kernel_rx_value(kernel, 0.33, np.pi)
        check_kernel_rx_value(kernel, np.pi / 2, np.pi)
        check_kernel_rx_value(kernel, 0, 0.55)
        check_kernel_rx_value(kernel, 0.33, 0.55)
        check_kernel_rx_value(kernel, np.pi / 2, 0.55)
