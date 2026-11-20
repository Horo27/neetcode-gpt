import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        res = {}

        x = np.array(x)
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)

        z1 = np.dot(x, W1.T) + b1
        a1 = np.maximum(0, z1)

        z2 = np.dot(a1, W2.T) + b2
        y_pred = z2

        loss = np.mean((y_pred - y_true) ** 2)

        res['loss'] = round(float(loss), 4)
        
        dl_dz2 = 2 / len(y_pred) * (y_pred - y_true)

        dl_dw2 = np.outer(dl_dz2, a1)

        dl_db2 = dl_dz2
    
        res['dW2'] = [[round(float(v), 4) for v in row] for row in dl_dw2]
        res['db2'] = [round(float(dl_db2_i), 4) for dl_db2_i in dl_db2]

        dl_da1 = np.dot(dl_dz2, W2)
        da1_dz1 = (z1 > 0).astype(float)

        dl_dz1 = dl_da1 * da1_dz1

        dl_dw1 = np.outer(dl_dz1, x)
        dl_db1 = dl_dz1

        res['dW1'] = [[round(float(v), 4) for v in row] for row in dl_dw1]
        res['db1'] = [round(float(v), 4) for v in dl_db1]

        return res

        
