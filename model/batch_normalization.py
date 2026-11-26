import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        x = np.asarray(x, dtype=float)                    # (N, D)
        gamma = np.asarray(gamma, dtype=float)            # (D,)
        beta = np.asarray(beta, dtype=float)              # (D,)
        running_mean = np.asarray(running_mean, dtype=float)
        running_var = np.asarray(running_var, dtype=float)

        mean = x.mean(axis = 0)
        var = x.var(axis = 0)
        if training:
            x_hat = (x - mean) / np.sqrt(var + eps)

            running_mean  = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var
        
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        
        y = gamma * x_hat + beta

        return [np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4)]
