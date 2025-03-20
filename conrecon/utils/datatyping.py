import numpy as np
from dataclasses import dataclass
"""
This class is solely for pickling the pca benchmark resutls and make data hanling easier wherever it is loaded.
"""
@dataclass
class PCABenchmarkResults():
    m1_utilities: list[float]
    m2_utilities: list[float]
    m1_privacies: list[float]
    m2_privacies: list[float]
    m1_m2_num_removed_components: list[int]
    num_iterations: int

