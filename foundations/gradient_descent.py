class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        
        def update(x, lr):
            return x - lr * 2 * x

        for i in range(0, iterations):
            init = update(init, learning_rate)
        return round(init, 5)
