import argparse
from typing import List

import control as ct
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-s", "--state_size", default=6, help="How many systems to generate"
    )
    ap.add_argument(
        "-n", "--num_of_gens", default=6, help="How many systems to generate"
    )
    ap.add_argument(
        "-i", "--num_inputs", default=3, help="How many input/outs the system ha."
    )
    ap.add_argument(
        "-o", "--num_outputs", default=1, help="How many input/outs the system ha."
    )
    ap.add_argument("--no-autoindent")
    # manage ipython indent argument
    return ap.parse_args()


def generate_matrix_A(amount: int, dim: int) -> list[np.ndarray]:
    # TODO: Ensure atleast one positive eigenvalue
    # For instability
    # rand_eigenvals = np.random.uniform(0, 1, (amount, dim))
    # For stability
    rand_eigenvals = np.random.uniform(-0.99, 0, (amount, dim))

    As = []
    for i in range(amount):
        A = np.diag(rand_eigenvals[i, :])
        P = np.random.rand(A.shape[0], A.shape[0])
        A = np.dot(np.dot(np.linalg.inv(P), A), P)
        As.append(A)
    return As


def generate_controllable_system(
    Amats: List[np.ndarray], num_inputs
) -> List[np.ndarray]:
    """
    Exsure controllability_matrix
    """
    Bs = []
    for ai in range(len(Amats)):
        n = Amats[ai].shape[0]
        B = np.random.rand(n, num_inputs)
        controllability_matrix = np.hstack(
            [np.linalg.matrix_power(Amats[ai], i).dot(B) for i in range(n)]
        )
        while matrix_rank(controllability_matrix) < n:
            B = np.random.rand(n, num_inputs)
            controllability_matrix = np.hstack(
                [np.linalg.matrix_power(Amats[ai], i).dot(B) for i in range(n)]
            )
        Bs.append(B)
    return Bs


def generate_state_space_systems(
    n, num_inputs, num_outputs, state_size, amount_systems
):
    print(f"Generating {amount_systems} systems")
    print("Generatinng Matrices A...")
    Amats = generate_matrix_A(amount_systems, state_size)
    print("Generatinng Matrices B...")
    Bmats = generate_controllable_system(Amats, num_inputs)
    print("Generatinng Matrices C...")
    Cmats = [np.random.rand(num_outputs, state_size) for i in range(amount_systems)]
    print("Generatinng Matrices D...")
    Dmats = [np.zeros((num_outputs, num_inputs)) for _ in range(amount_systems)]
    return Amats, Bmats, Cmats, Dmats


if __name__ == "__main__":
    args = argsies()
    # Get out matrices
    Amats, Bmats, Cmats, Dmats = generate_state_space_systems(
        args.state_size,
        args.num_inputs,
        args.num_outputs,
        args.state_size,
        args.num_of_gens,
    )

    ct.use_fbs_defaults()  # Use settings to match FBS
    for i in range(args.num_of_gens):
        A, B, C = Amats[i], Bmats[i], Cmats[i]
        sys = ct.ss(A, B, C, 0)

        t = np.linspace(0, 10, 101)
        u = np.zeros((args.num_inputs, len(t)))
        print(f"T is of shape {t.shape} with length {len(t)}")
        # u[:, int(len(t) / 4)] = 1
        init_cond = np.random.uniform(0, 1, sys.nstates)
        # Lets run the simulation and see what happens
        timepts = np.linspace(0, 10, 101)
        response = ct.input_output_response(sys, timepts, u, X0=init_cond)
        print(f"Generation {i} response: {response.outputs}")
        # Lets plot the response
        plt.plot(timepts, response.outputs.squeeze())
        plt.show()
    print(f"Generating {args.num_of_gens} generations")
