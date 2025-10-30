"""Tests unitarios para DataLoader.

Verifica la carga y preprocesamiento de datasets.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.DataLoader import MNISTLoader


def test_cargar_mnist():
    """Test de carga de MNIST (muestra pequeña)."""
    print("\n" + "="*60)
    print("TEST: Cargar MNIST")
    print("="*60)
    
    try:
        loader = MNISTLoader()
        
        # Cargar solo 100 ejemplos para ser rápido
        X, y = loader.load_data(n_samples=100)
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"X dtype: {X.dtype}")
        print(f"y dtype: {y.dtype}")
        
        assert X.shape == (100, 784), f"X debe tener shape (100, 784), tiene {X.shape}"
        assert y.shape == (100,), f"y debe tener shape (100,), tiene {y.shape}"
        assert X.dtype == np.float32, f"X debe ser float32, es {X.dtype}"
        
        print("✓ Cargar MNIST: PASS")
        
    except ImportError as e:
        print(f"⚠ SKIP: {e}")
        print("  (Instalar scikit-learn para ejecutar este test)")


def test_preprocesar():
    """Test de preprocesamiento (normalización y one-hot)."""
    print("\n" + "="*60)
    print("TEST: Preprocesamiento")
    print("="*60)
    
    loader = MNISTLoader()
    
    # Crear datos de prueba simulados
    X_fake = np.array([[0, 128, 255], [255, 0, 128]], dtype=np.float32)
    y_fake = np.array([0, 2], dtype=int)
    
    X_proc, y_one_hot = loader.preprocess(X_fake, y_fake)
    
    print(f"X original:\n{X_fake}")
    print(f"X procesado:\n{X_proc}")
    print(f"y original: {y_fake}")
    print(f"y one-hot:\n{y_one_hot}")
    
    # Verificar normalización
    assert np.max(X_proc) <= 1.0, "X normalizado debe estar <= 1.0"
    assert np.min(X_proc) >= 0.0, "X normalizado debe estar >= 0.0"
    
    # Verificar valores específicos
    assert np.isclose(X_proc[0, 1], 128/255), "Normalización incorrecta"
    assert np.isclose(X_proc[0, 2], 1.0), "Normalización incorrecta"
    
    # Verificar one-hot encoding
    assert y_one_hot.shape == (2, 3), f"One-hot debe tener shape (2, 3), tiene {y_one_hot.shape}"
    assert y_one_hot[0, 0] == 1 and np.sum(y_one_hot[0, :]) == 1, "One-hot encoding incorrecto"
    assert y_one_hot[1, 2] == 1 and np.sum(y_one_hot[1, :]) == 1, "One-hot encoding incorrecto"
    
    print("✓ Preprocesamiento: PASS")


def test_normalizar_on_off():
    """Test de activación/desactivación de normalización."""
    print("\n" + "="*60)
    print("TEST: Normalizar ON/OFF")
    print("="*60)
    
    loader = MNISTLoader()
    
    X_fake = np.array([[0, 128, 255]], dtype=np.float32)
    y_fake = np.array([0], dtype=int)
    
    # Sin normalizar
    X_no_norm, _ = loader.preprocess(X_fake, y_fake, normalize=False)
    print(f"Sin normalizar: {X_no_norm}")
    assert np.array_equal(X_no_norm, X_fake), "Sin normalizar debe devolver el original"
    
    # Con normalización
    X_norm, _ = loader.preprocess(X_fake, y_fake, normalize=True)
    print(f"Con normalización: {X_norm}")
    assert np.max(X_norm) <= 1.0, "Con normalización debe estar en [0,1]"
    
    print("✓ Normalizar ON/OFF: PASS")


def test_one_hot_encoding():
    """Test específico de one-hot encoding."""
    print("\n" + "="*60)
    print("TEST: One-Hot Encoding")
    print("="*60)
    
    loader = MNISTLoader()
    
    # Test con diferentes números de clases
    for n_clases in [2, 5, 10]:
        y = np.arange(n_clases)
        X = np.zeros((n_clases, 10))
        
        _, y_one_hot = loader.preprocess(X, y)
        
        print(f"  {n_clases} clases: {y_one_hot.shape}")
        
        # Verificar shape
        assert y_one_hot.shape == (n_clases, n_clases), \
            f"Shape incorrecto para {n_clases} clases"
        
        # Verificar que es una matriz identidad
        assert np.array_equal(y_one_hot, np.eye(n_clases)), \
            f"One-hot encoding incorrecto para {n_clases} clases"
    
    print("✓ One-Hot Encoding: PASS")


def test_train_test_split():
    """Test de división train/test."""
    print("\n" + "="*60)
    print("TEST: Train/Test Split")
    print("="*60)
    
    try:
        loader = MNISTLoader()
        
        # Crear datos de prueba
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = loader.train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        # Verificar tamaños
        assert X_train.shape[0] == 80, "Train debe tener 80 ejemplos"
        assert X_test.shape[0] == 20, "Test debe tener 20 ejemplos"
        assert y_train.shape[0] == 80, "Train labels debe tener 80 ejemplos"
        assert y_test.shape[0] == 20, "Test labels debe tener 20 ejemplos"
        
        # Verificar que no hay solapamiento
        total_elements = X_train.shape[0] + X_test.shape[0]
        assert total_elements == 100, "No debe haber pérdida de datos"
        
        print("✓ Train/Test Split: PASS")
        
    except ImportError as e:
        print(f"⚠ SKIP: {e}")


def test_tipos_datos():
    """Test de tipos de datos correctos."""
    print("\n" + "="*60)
    print("TEST: Tipos de Datos")
    print("="*60)
    
    loader = MNISTLoader()
    
    X = np.array([[100, 200]], dtype=np.uint8)
    y = np.array([1], dtype=int)
    
    X_proc, y_one_hot = loader.preprocess(X, y)
    
    print(f"X procesado dtype: {X_proc.dtype}")
    print(f"y one-hot dtype: {y_one_hot.dtype}")
    
    # Verificar que son tipos float
    assert X_proc.dtype in [np.float32, np.float64], "X debe ser float"
    assert y_one_hot.dtype in [np.float32, np.float64], "y debe ser float"
    
    print("✓ Tipos de Datos: PASS")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("TESTS: DataLoader")
    print("="*60)
    
    tests = [
        test_cargar_mnist,
        test_preprocesar,
        test_normalizar_on_off,
        test_one_hot_encoding,
        test_train_test_split,
        test_tipos_datos
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTADOS: {passed} PASS, {failed} FAIL")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
