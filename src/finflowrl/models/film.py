"""FiLM — Feature-wise Linear Modulation layer.

FiLM applies an affine transformation conditioned on auxiliary features:
    output = gamma * input + beta

where gamma and beta are projected from the conditioning vector.
Used to condition the MeanFlow policy on market state.
"""

import numpy as np


class FiLMLayer:
    """Feature-wise Linear Modulation layer.

    Projects conditioning input into scale (gamma) and shift (beta)
    parameters, then applies element-wise affine transform to the input.
    """

    def __init__(self, input_dim: int, cond_dim: int):
        """
        Args:
            input_dim: dimension of the input to modulate
            cond_dim: dimension of the conditioning vector
        """
        # Project conditioning to gamma and beta
        scale = np.sqrt(2.0 / cond_dim)
        self.W_gamma = np.random.randn(cond_dim, input_dim) * scale
        self.b_gamma = np.ones(input_dim)
        self.W_beta = np.random.randn(cond_dim, input_dim) * scale
        self.b_beta = np.zeros(input_dim)

        self.input_dim = input_dim
        self.cond_dim = cond_dim

    def __call__(self, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """Apply FiLM modulation.

        Args:
            x: input tensor of shape (..., input_dim)
            cond: conditioning vector of shape (..., cond_dim)

        Returns:
            modulated tensor of shape (..., input_dim)
        """
        gamma = cond @ self.W_gamma + self.b_gamma
        beta = cond @ self.W_beta + self.b_beta
        return gamma * x + beta

    def get_params(self) -> dict:
        """Return parameters as dict."""
        return {
            "W_gamma": self.W_gamma.copy(),
            "b_gamma": self.b_gamma.copy(),
            "W_beta": self.W_beta.copy(),
            "b_beta": self.b_beta.copy(),
        }

    def set_params(self, params: dict) -> None:
        """Set parameters from dict."""
        self.W_gamma = params["W_gamma"]
        self.b_gamma = params["b_gamma"]
        self.W_beta = params["W_beta"]
        self.b_beta = params["b_beta"]
