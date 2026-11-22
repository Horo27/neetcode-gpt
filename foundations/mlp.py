import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

        def ReLu(x):
            return [max(0, i) for i in x]

        curr_output = x

        for i in range(len(weights) - 1):
            z = np.dot(curr_output, weights[i]) + biases[i]
            print(z)
            curr_output = ReLu(z)
        return np.round(np.dot(curr_output, weights[-1]) + biases[-1], 5)
