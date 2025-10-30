"""Tests unitarios para el Distorsionador.

Verifica la generación correcta de datasets con distorsión.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tempfile
import shutil
from src.Distorsionador import Distorsionador


def test_cargar_patrones_base():
    """Test de carga de patrones base."""
    print("\n" + "="*60)
    print("TEST: Cargar Patrones Base")
    print("="*60)
    
    dist = Distorsionador(seed=42)
    
    # Cargar desde el dataset original
    patrones = dist.cargar_patrones_base("datasets/originales/100/letras.csv")
    
    print(f"Patrones cargados: {len(patrones)}")
    
    assert len(patrones) > 0, "Debe cargar al menos 1 patrón"
    assert all(len(p) == 2 for p in patrones), "Cada patrón debe tener (features, label)"
    
    # Verificar que los patrones tienen 102 features
    for patron, label in patrones:
        assert len(patron) == 102, f"Patrón debe tener 102 features, tiene {len(patron)}"
        assert label in [0, 1], f"Label debe ser 0 o 1, es {label}"
    
    print("✓ Cargar Patrones Base: PASS")


def test_distorsionar_patron():
    """Test de distorsión de un patrón individual."""
    print("\n" + "="*60)
    print("TEST: Distorsionar Patrón")
    print("="*60)
    
    dist = Distorsionador(seed=42)
    
    # Crear un patrón de prueba
    patron_original = np.zeros(102, dtype=int)
    patron_original[0:20] = 1  # 20 píxeles en 1
    
    # Aplicar 10% de distorsión (aproximadamente 10 píxeles)
    patron_distorsionado = dist.distorsionar_patron(patron_original, 10.0)
    
    # Contar diferencias
    diferencias = np.sum(patron_original != patron_distorsionado)
    print(f"Píxeles modificados: {diferencias} / {len(patron_original)}")
    print(f"Porcentaje real: {(diferencias / len(patron_original)) * 100:.1f}%")
    
    # Verificar que se modificó aproximadamente el 10%
    esperado = int(np.ceil(len(patron_original) * 0.10))
    assert diferencias == esperado, f"Debería modificar {esperado} píxeles, modificó {diferencias}"
    
    print("✓ Distorsionar Patrón: PASS")


def test_distorsion_rangos():
    """Test de distorsión en diferentes rangos."""
    print("\n" + "="*60)
    print("TEST: Rangos de Distorsión")
    print("="*60)
    
    dist = Distorsionador(seed=42)
    patron_base = np.zeros(102, dtype=int)
    patron_base[0:50] = 1
    
    rangos = [1.0, 10.0, 20.0, 30.0]
    
    for porcentaje in rangos:
        patron_dist = dist.distorsionar_patron(patron_base, porcentaje)
        diferencias = np.sum(patron_base != patron_dist)
        esperado = int(np.ceil(102 * porcentaje / 100.0))
        
        print(f"  Distorsión {porcentaje}%: {diferencias} píxeles (esperado: {esperado})")
        assert diferencias == esperado, f"Distorsión {porcentaje}% incorrecta"
    
    print("✓ Rangos de Distorsión: PASS")


def test_generar_dataset_cantidad():
    """Test de generación con la cantidad correcta de ejemplos."""
    print("\n" + "="*60)
    print("TEST: Cantidad de Ejemplos")
    print("="*60)
    
    dist = Distorsionador(seed=42)
    dist.cargar_patrones_base("datasets/originales/100/letras.csv")
    
    cantidades = [10, 50, 100]
    
    for n in cantidades:
        X, y = dist.generar_dataset(n_ejemplos=n)
        
        print(f"  Dataset {n}: X.shape={X.shape}, y.shape={y.shape}")
        
        assert X.shape[0] == n, f"Debe generar {n} ejemplos, generó {X.shape[0]}"
        assert y.shape[0] == n, f"Debe generar {n} labels, generó {y.shape[0]}"
        assert X.shape[1] == 102, f"Cada ejemplo debe tener 102 features"
    
    print("✓ Cantidad de Ejemplos: PASS")


def test_distribucion_10_90():
    """Test de distribución 10% original - 90% distorsionado."""
    print("\n" + "="*60)
    print("TEST: Distribución 10% / 90%")
    print("="*60)
    
    dist = Distorsionador(seed=42)
    dist.cargar_patrones_base("datasets/originales/100/letras.csv")
    
    n_ejemplos = 100
    X, y = dist.generar_dataset(n_ejemplos=n_ejemplos)
    
    # Extraer patrones base
    patrones_base = [patron for patron, label in dist.patrones_base]
    
    # Contar cuántos ejemplos son idénticos a algún patrón base
    sin_distorsion = 0
    for ejemplo in X:
        for patron_base in patrones_base:
            if np.array_equal(ejemplo, patron_base):
                sin_distorsion += 1
                break
    
    con_distorsion = n_ejemplos - sin_distorsion
    porcentaje_sin = (sin_distorsion / n_ejemplos) * 100
    porcentaje_con = (con_distorsion / n_ejemplos) * 100
    
    print(f"  Sin distorsión: {sin_distorsion} ({porcentaje_sin:.1f}%)")
    print(f"  Con distorsión: {con_distorsion} ({porcentaje_con:.1f}%)")
    
    # Verificar que está cerca del 10% / 90%
    assert 8 <= porcentaje_sin <= 12, f"Debe tener ~10% sin distorsión, tiene {porcentaje_sin:.1f}%"
    assert 88 <= porcentaje_con <= 92, f"Debe tener ~90% con distorsión, tiene {porcentaje_con:.1f}%"
    
    print("✓ Distribución 10% / 90%: PASS")


def test_distribucion_clases():
    """Test de distribución de clases."""
    print("\n" + "="*60)
    print("TEST: Distribución de Clases")
    print("="*60)
    
    dist = Distorsionador(seed=42)
    dist.cargar_patrones_base("datasets/originales/100/letras.csv")
    
    # Generar con distribución específica
    distribucion = {0: 0.7, 1: 0.3}
    X, y = dist.generar_dataset(
        n_ejemplos=100,
        distribucion_clases=distribucion
    )
    
    # Contar clases
    conteo = np.bincount(y)
    print(f"  Distribución obtenida: {conteo}")
    print(f"  Proporción: {conteo / len(y)}")
    
    # Verificar aproximadamente 70-30
    assert 60 <= conteo[0] <= 80, f"Clase 0 debe estar cerca de 70%, tiene {conteo[0]}"
    assert 20 <= conteo[1] <= 40, f"Clase 1 debe estar cerca de 30%, tiene {conteo[1]}"
    
    print("✓ Distribución de Clases: PASS")


def test_guardar_dataset():
    """Test de guardado de dataset en archivo."""
    print("\n" + "="*60)
    print("TEST: Guardar Dataset")
    print("="*60)
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    
    try:
        dist = Distorsionador(seed=42)
        dist.cargar_patrones_base("datasets/originales/100/letras.csv")
        
        output_path = os.path.join(temp_dir, "test_dataset.csv")
        X, y = dist.generar_dataset(n_ejemplos=50, output_path=output_path)
        
        # Verificar que el archivo existe
        assert os.path.exists(output_path), "El archivo debe existir"
        
        # Recargar y verificar
        data = np.loadtxt(output_path, delimiter=';', dtype=int)
        
        print(f"  Archivo guardado: {output_path}")
        print(f"  Shape del archivo: {data.shape}")
        
        assert data.shape == (50, 103), f"Archivo debe tener shape (50, 103), tiene {data.shape}"
        
        # Verificar que los datos coinciden
        X_reloaded = data[:, :-1]
        y_reloaded = data[:, -1]
        
        assert np.array_equal(X_reloaded, X), "Features no coinciden"
        assert np.array_equal(y_reloaded, y), "Labels no coinciden"
        
        print("✓ Guardar Dataset: PASS")
        
    finally:
        # Limpiar directorio temporal
        shutil.rmtree(temp_dir)


def test_reproducibilidad():
    """Test de reproducibilidad con mismo seed."""
    print("\n" + "="*60)
    print("TEST: Reproducibilidad")
    print("="*60)
    
    # Generar dos veces con el mismo seed
    dist1 = Distorsionador(seed=123)
    dist1.cargar_patrones_base("datasets/originales/100/letras.csv")
    X1, y1 = dist1.generar_dataset(n_ejemplos=50)
    
    dist2 = Distorsionador(seed=123)
    dist2.cargar_patrones_base("datasets/originales/100/letras.csv")
    X2, y2 = dist2.generar_dataset(n_ejemplos=50)
    
    # Verificar que son idénticos
    assert np.array_equal(X1, X2), "X debe ser idéntico con mismo seed"
    assert np.array_equal(y1, y2), "y debe ser idéntico con mismo seed"
    
    print("✓ Reproducibilidad: PASS")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("TESTS: Distorsionador")
    print("="*60)
    
    tests = [
        test_cargar_patrones_base,
        test_distorsionar_patron,
        test_distorsion_rangos,
        test_generar_dataset_cantidad,
        test_distribucion_10_90,
        test_distribucion_clases,
        test_guardar_dataset,
        test_reproducibilidad
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
