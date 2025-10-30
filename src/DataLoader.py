"""Carga y preprocesamiento de MNIST usando sklearn (fetch_openml).

Provee funciones para cargar, normalizar y devolver splits train/test.
"""
import numpy as np

try:
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
except Exception:
    fetch_openml = None
    train_test_split = None


class MNISTLoader:
    def __init__(self):
        self.X = None
        self.y = None

    def load_data(self, n_samples=None, as_float32=True):
        """Carga MNIST. Si sklearn no está instalado, lanza una excepción clara.

        n_samples: si se pasa, limita la cantidad de ejemplos (útil para pruebas rápidas).
        """
        if fetch_openml is None:
            raise ImportError("scikit-learn no está disponible. Instala scikit-learn para cargar MNIST.")

        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X = mnist.data
        y = mnist.target.astype(int)

        if n_samples is not None:
            X = X[:n_samples]
            y = y[:n_samples]

        if as_float32:
            X = X.astype('float32')

        self.X = X
        self.y = y
        return X, y

    def preprocess(self, X, y, normalize=True):
        """Normaliza los píxeles a [0,1] y transforma las etiquetas a one-hot.

        Retorna (X_proc, y_one_hot)
        """
        if normalize:
            X_proc = X / 255.0
        else:
            X_proc = X.copy()

        # one-hot
        n_classes = int(np.max(y)) + 1
        y_one_hot = np.zeros((y.shape[0], n_classes), dtype=float)
        y_one_hot[np.arange(y.shape[0]), y] = 1.0

        return X_proc, y_one_hot

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        if train_test_split is None:
            raise ImportError("scikit-learn no está disponible. Instala scikit-learn para usar train_test_split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    loader = MNISTLoader()
    X, y = loader.load_data(n_samples=1000)
    Xp, y1 = loader.preprocess(X, y)
    print("X shape:", Xp.shape)
    print("y one-hot shape:", y1.shape)
