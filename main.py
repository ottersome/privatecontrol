import argparse

import control as ct
import matplotlib.pyplot as plt
import numpy as np

from conrecon.automated_generation import generate_state_space_systems


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

    # CHECK: Wy necessary
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
