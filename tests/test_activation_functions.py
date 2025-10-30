"""Tests unitarios para las funciones de activación.

Verifica que todas las funciones de activación y sus derivadas
funcionen correctamente con diferentes inputs.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.ActivationFunctions import ActivationFunction


def test_relu():
    """Test de la función ReLU."""
    print("\n" + "="*60)
    print("TEST: ReLU")
    print("="*60)
    
    # Test con valores negativos, cero y positivos
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
    
    result = ActivationFunction.relu(x)
    
    print(f"Input:    {x}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    
    assert np.allclose(result, expected), "ReLU falló"
    print("✓ ReLU: PASS")


def test_relu_derivative():
    """Test de la derivada de ReLU."""
    print("\n" + "="*60)
    print("TEST: ReLU Derivative")
    print("="*60)
    
    x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    
    result = ActivationFunction.relu_derivative(x)
    
    print(f"Input:    {x}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    
    assert np.allclose(result, expected), "ReLU derivative falló"
    print("✓ ReLU Derivative: PASS")


def test_sigmoid():
    """Test de la función Sigmoid."""
    print("\n" + "="*60)
    print("TEST: Sigmoid")
    print("="*60)
    
    x = np.array([[0.0, 1.0, -1.0]])
    result = ActivationFunction.sigmoid(x)
    
    print(f"Input:  {x}")
    print(f"Output: {result}")
    
    # Verificar propiedades de sigmoid
    assert np.all(result >= 0) and np.all(result <= 1), "Sigmoid debe estar en [0,1]"
    assert np.isclose(result[0, 0], 0.5), "sigmoid(0) debe ser 0.5"
    
    print("✓ Sigmoid: PASS")


def test_sigmoid_derivative():
    """Test de la derivada de Sigmoid."""
    print("\n" + "="*60)
    print("TEST: Sigmoid Derivative")
    print("="*60)
    
    x = np.array([[0.0]])
    result = ActivationFunction.sigmoid_derivative(x)
    expected = 0.25  # sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
    
    print(f"Input:    {x}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    
    assert np.isclose(result[0, 0], expected), "Sigmoid derivative en 0 debe ser 0.25"
    print("✓ Sigmoid Derivative: PASS")


def test_tanh():
    """Test de la función Tanh."""
    print("\n" + "="*60)
    print("TEST: Tanh")
    print("="*60)
    
    x = np.array([[0.0, 1.0, -1.0, 10.0, -10.0]])
    result = ActivationFunction.tanh(x)
    
    print(f"Input:  {x}")
    print(f"Output: {result}")
    
    # Verificar propiedades de tanh
    assert np.all(result >= -1) and np.all(result <= 1), "Tanh debe estar en [-1,1]"
    assert np.isclose(result[0, 0], 0.0), "tanh(0) debe ser 0"
    assert result[0, 1] > 0 and result[0, 2] < 0, "tanh debe ser simétrica"
    
    print("✓ Tanh: PASS")


def test_tanh_derivative():
    """Test de la derivada de Tanh."""
    print("\n" + "="*60)
    print("TEST: Tanh Derivative")
    print("="*60)
    
    x = np.array([[0.0]])
    result = ActivationFunction.tanh_derivative(x)
    expected = 1.0  # tanh'(0) = 1 - 0^2 = 1
    
    print(f"Input:    {x}")
    print(f"Output:   {result}")
    print(f"Expected: {expected}")
    
    assert np.isclose(result[0, 0], expected), "Tanh derivative en 0 debe ser 1.0"
    print("✓ Tanh Derivative: PASS")


def test_softmax():
    """Test de la función Softmax."""
    print("\n" + "="*60)
    print("TEST: Softmax")
    print("="*60)
    
    # Test con 2 ejemplos, 3 clases cada uno
    x = np.array([[1.0, 2.0, 3.0],
                  [1.0, 1.0, 1.0]])
    result = ActivationFunction.softmax(x)
    
    print(f"Input:\n{x}")
    print(f"Output:\n{result}")
    
    # Verificar que suma 1 por fila
    sumas = np.sum(result, axis=1)
    print(f"Sumas por fila: {sumas}")
    
    assert np.allclose(sumas, 1.0), "Softmax debe sumar 1 por fila"
    assert np.all(result >= 0) and np.all(result <= 1), "Softmax debe estar en [0,1]"
    
    # Para entradas iguales, probabilidades deben ser iguales
    assert np.allclose(result[1], [1/3, 1/3, 1/3]), "Softmax con inputs iguales debe dar prob uniformes"
    
    print("✓ Softmax: PASS")


def test_numerical_stability():
    """Test de estabilidad numérica con valores extremos."""
    print("\n" + "="*60)
    print("TEST: Estabilidad Numérica")
    print("="*60)
    
    # Valores muy grandes y muy pequeños
    x_extreme = np.array([[1000.0, -1000.0, 0.0]])
    
    # Sigmoid debe manejar valores extremos
    sigmoid_result = ActivationFunction.sigmoid(x_extreme)
    assert not np.any(np.isnan(sigmoid_result)), "Sigmoid no debe producir NaN"
    assert not np.any(np.isinf(sigmoid_result)), "Sigmoid no debe producir Inf"
    print("✓ Sigmoid estable con valores extremos")
    
    # ReLU debe manejar valores extremos
    relu_result = ActivationFunction.relu(x_extreme)
    assert not np.any(np.isnan(relu_result)), "ReLU no debe producir NaN"
    print("✓ ReLU estable con valores extremos")
    
    # Softmax debe manejar valores extremos
    x_softmax = np.array([[1000.0, 500.0, 0.0]])
    softmax_result = ActivationFunction.softmax(x_softmax)
    assert not np.any(np.isnan(softmax_result)), "Softmax no debe producir NaN"
    assert not np.any(np.isinf(softmax_result)), "Softmax no debe producir Inf"
    assert np.isclose(np.sum(softmax_result), 1.0), "Softmax debe sumar 1"
    print("✓ Softmax estable con valores extremos")
    
    print("\n✓ Estabilidad Numérica: PASS")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("TESTS: ActivationFunctions")
    print("="*60)
    
    tests = [
        test_relu,
        test_relu_derivative,
        test_sigmoid,
        test_sigmoid_derivative,
        test_tanh,
        test_tanh_derivative,
        test_softmax,
        test_numerical_stability
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
