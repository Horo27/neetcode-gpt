import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        
        res = 0
        eps = 10**(-7)

        for yt, yp in zip(y_true, y_pred):
            if yp == 0 or yp == 1:
                yp += eps

            res += (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))

        return round(res * (-1 / len(y_true)), 4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        res = 0
        eps = 10**(-7)

        for yt, yp in zip(y_true, y_pred):
            for t, p in zip(yt, yp):
                if p == 0:
                    p += eps
                print(t, " * ", "log(",p,")")
                res += t * np.log(p)
        print(len(y_true[0]))
        return round(res * (-1 / len(y_true)), 4)






