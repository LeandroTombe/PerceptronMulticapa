"""Script para comparar datasets originales vs distorsionados."""
import numpy as np
import os


def comparar_datasets(original_path, distorsionado_path, nombre):
    """Compara un dataset original con su versión distorsionada."""
    print(f"\n{'='*70}")
    print(f"COMPARACIÓN: {nombre}")
    print(f"{'='*70}")
    
    # Cargar ambos datasets
    original = np.loadtxt(original_path, delimiter=';', dtype=int)
    distorsionado = np.loadtxt(distorsionado_path, delimiter=';', dtype=int)
    
    # Separar features y labels
    X_orig = original[:, :-1]
    y_orig = original[:, -1]
    X_dist = distorsionado[:, :-1]
    y_dist = distorsionado[:, -1]
    
    print(f"\n📊 ORIGINAL:")
    print(f"   Ejemplos: {X_orig.shape[0]}")
    print(f"   Patrones únicos: {len(np.unique(X_orig, axis=0))}")
    print(f"   Distribución clases: {np.bincount(y_orig)}")
    
    print(f"\n📊 DISTORSIONADO:")
    print(f"   Ejemplos: {X_dist.shape[0]}")
    print(f"   Patrones únicos: {len(np.unique(X_dist, axis=0))}")
    print(f"   Distribución clases: {np.bincount(y_dist)}")
    
    # Calcular diferencias promedio
    # Nota: no podemos comparar 1 a 1 porque están mezclados
    print(f"\n📈 ESTADÍSTICAS:")
    densidad_orig = np.mean(X_orig) * 100
    densidad_dist = np.mean(X_dist) * 100
    print(f"   Densidad original: {densidad_orig:.2f}%")
    print(f"   Densidad distorsionada: {densidad_dist:.2f}%")
    print(f"   Cambio en densidad: {abs(densidad_dist - densidad_orig):.2f}%")
    
    # Verificar ejemplos sin distorsión
    # Buscar cuántos ejemplos del dataset distorsionado son idénticos a los patrones base
    patrones_base = np.unique(X_orig, axis=0)
    sin_distorsion = 0
    for patron in X_dist:
        for base in patrones_base:
            if np.array_equal(patron, base):
                sin_distorsion += 1
                break
    
    porcentaje_sin_dist = (sin_distorsion / len(X_dist)) * 100
    print(f"\n✓ VERIFICACIÓN REQUISITOS TP:")
    print(f"   Ejemplos sin distorsión: {sin_distorsion} ({porcentaje_sin_dist:.1f}%)")
    print(f"   Ejemplos con distorsión: {len(X_dist) - sin_distorsion} ({100-porcentaje_sin_dist:.1f}%)")
    
    if 8 <= porcentaje_sin_dist <= 12:
        print(f"   ✓ CUMPLE: ~10% sin distorsión")
    else:
        print(f"   ⚠ Fuera de rango esperado (8-12%)")


def main():
    print("\n" + "="*70)
    print("VERIFICACIÓN DE DATASETS DISTORSIONADOS")
    print("="*70)
    
    configs = [
        ("100", "datasets/originales/100/letras.csv", "datasets/distorsionados/100/letras.csv"),
        ("500", "datasets/originales/500/letras.csv", "datasets/distorsionados/500/letras.csv"),
        ("1000", "datasets/originales/1000/letras.csv", "datasets/distorsionados/1000/letras.csv")
    ]
    
    for nombre, orig, dist in configs:
        if os.path.exists(orig) and os.path.exists(dist):
            comparar_datasets(orig, dist, f"{nombre} ejemplos")
        else:
            print(f"\n⚠ No se encontraron archivos para: {nombre}")
    
    print("\n" + "="*70)
    print("✓ VERIFICACIÓN COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    main()
