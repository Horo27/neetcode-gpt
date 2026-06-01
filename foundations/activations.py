import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = []
        for x in z:
            arr.append(round(1 / (1 + np.e ** (-x)), 5))
        return np.array(arr)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = []
        for x in z:
            arr.append(max(0.0, x))
        return np.array(arr)
