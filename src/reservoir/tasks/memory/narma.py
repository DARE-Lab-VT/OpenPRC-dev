import numpy as np
from ..base import BaseTask

class NARMA(BaseTask):
    """
    NARMA task matching the reference logic.
    Supports N=2 (Simpler cubic) and N>2 (Standard NARMA).
    """
    def __init__(self, order=2, a=0.3, b=0.05, c=1.5, d=0.1):
        self.order = order
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def generate(self, u_input):
        """
        Args:
            u_input (np.ndarray, optional): External input signal driving the reservoir.
                                            If None, generates random uniform [0, 0.5].
        """
        length = len(u_input)
        # 1. Handle Input Signal
        # Ensure input matches requested length
        if len(u_input) != length:
            raise ValueError(f"Input u length ({len(u_input)}) does not match requested length ({length}).")
        u = u_input.flatten()
        
        y = np.zeros(length)
        N = self.order
        
        # 2. Logic matching target_narma from reference
        if N == 2:
            # N=2 Special Case (Cubic nonlinearity)
            for t in range(2, length):
                y[t] = 0.4 * y[t - 1] + \
                       0.4 * y[t - 1] * y[t - 2] + \
                       0.6 * u[t - 1] ** 3 + \
                       0.1
        else:
            # Standard NARMA Case
            for t in range(N, length):
                sum_y = np.sum(y[t - N:t])
                y[t] = self.a * y[t - 1] + \
                       self.b * y[t - 1] * sum_y + \
                       self.c * u[t - N] * u[t - 1] + \
                       self.d
                       
        return y.reshape(-1, 1)