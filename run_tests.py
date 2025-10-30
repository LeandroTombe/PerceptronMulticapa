"""Script para ejecutar todos los tests del proyecto.

Ejecuta los tests de todos los módulos y genera un reporte completo.
"""
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests import test_activation_functions, test_distorsionador, test_dataloader


def print_header(title):
    """Imprime un encabezado formateado."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70)


def main():
    """Ejecuta todos los tests y genera un reporte."""
    
    print_header("SUITE COMPLETA DE TESTS - PERCEPTRÓN MULTICAPA")
    
    results = {}
    
    # Test 1: ActivationFunctions
    print("\n🧪 Ejecutando tests de ActivationFunctions...")
    results['ActivationFunctions'] = test_activation_functions.run_all_tests()
    
    # Test 2: Distorsionador
    print("\n🧪 Ejecutando tests de Distorsionador...")
    results['Distorsionador'] = test_distorsionador.run_all_tests()
    
    # Test 3: DataLoader
    print("\n🧪 Ejecutando tests de DataLoader...")
    results['DataLoader'] = test_dataloader.run_all_tests()
    
    # Reporte final
    print_header("REPORTE FINAL")
    
    total_passed = sum(1 for v in results.values() if v)
    total_failed = sum(1 for v in results.values() if not v)
    
    print("\n📊 Resultados por módulo:")
    for module, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {module:30s} {status}")
    
    print(f"\n📈 Resumen:")
    print(f"  Módulos PASS: {total_passed}")
    print(f"  Módulos FAIL: {total_failed}")
    
    if total_failed == 0:
        print("\n" + "🎉 " + "TODOS LOS TESTS PASARON".center(66) + " 🎉")
        print("="*70)
        return 0
    else:
        print("\n" + "⚠ " + f"ALGUNOS TESTS FALLARON ({total_failed})".center(66) + " ⚠")
        print("="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
