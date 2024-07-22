# figure-3-1-io_response.py - input/output response of a linear system
# RMM, 28 Aug 2021
#
# Figure 3.4: Input/output response of a linear system. The step
# response (a) shows the output of the system due to an input that
# changes from 0 to 1 at time t = 5 s. The frequency response (b)
# shows the amplitude gain and phase change due to a sinusoidal input
# at different frequencies.
#
import control as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy.random import uniform

ct.use_fbs_defaults()  # Use settings to match FBS

# System definition - third order, state space system
A = [[-0.2, 2, 0], [-0.5, -0.2, 4], [0, 0, -10]]
B = [0, 0, 1]
C = [2.6, 0, 0]
# Create a state space system
sys = ct.ss(A, B, C, 0)

# Set up the plotting grid to match the layout in the book
fig = plt.figure(constrained_layout=True, dpi=150, figsize=(15, 10))
gs = fig.add_gridspec(4, 2)

fig.add_subplot(gs[0:4, 0:2])  # first column

u[t < 5] = 0

# Set numpy seed
np.random.seed(420)

# Compute the response
# Random initial condition
initial_condition = np.random.rand(3)
response = ct.forced_response(sys, t, u, X0=initial_condition)
y = response.outputs
x = response.states
print(f"States are of shape {x.shape}")

# Plot the response
plt.plot(t, u, "b--", label="Input")
plt.plot(t, y, "r-", label="Output")
for s in x:
    plt.plot(t, s, "y-", label="States")
plt.xlabel("Time (sec)")
plt.ylabel("Input, output")
plt.title("Step response")
plt.title("States")
plt.legend()

plt.show()
#
# (b) Frequency` response showing the amplitude gain and phase change
# due to a sinusoidal input at different frequencies
#
#
# # Set up the axes for plotting (labels are recognized by bode_plot())
# mag = fig.add_subplot(gs[0, 1], label="Control-bode-magnitude")
# phase = fig.add_subplot(gs[1, 1], label="Control-bode-phase")
#
# # Generate the Bode plot
# ct.bode_plot(sys)
#
# # Adjust the appearance to match the book
# mag.xaxis.set_ticklabels([])
# mag.set_title("Frequency response")
#
# plt.show()
