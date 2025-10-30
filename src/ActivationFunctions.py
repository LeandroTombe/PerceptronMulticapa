"""Funciones de activación y sus derivadas.

Implementaciones vectorizadas pensadas para trabajar con lotes (batch) de forma
eficiente usando NumPy.
"""
import numpy as np


class ActivationFunction:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        # clipping para estabilidad numérica
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1.0 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z):
        # z shape: (batch_size, n_classes)
        shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shift)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


if __name__ == "__main__":
    # prueba rápida
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 0.5]])
    print("relu:\n", ActivationFunction.relu(x))
    print("sigmoid:\n", ActivationFunction.sigmoid(x))
    print("tanh:\n", ActivationFunction.tanh(x))
    print("softmax:\n", ActivationFunction.softmax(np.array([[1.0,2.0,3.0],[1.0,1.0,1.0]])))
