import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        z = z - max(z)
        sum_ = 0
        for i in z:
            sum_ += np.e**i
        arr = []        

        for i in z:
            arr.append(round((np.e**i) / sum_, 4))
        
        return np.array(arr)
        
        
