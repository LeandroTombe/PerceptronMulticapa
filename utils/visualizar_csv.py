"""
Script para visualizar los datasets CSV de forma legible
"""
import numpy as np
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def visualizar_patron(features, label, ancho=17, alto=6):
    """
    Visualiza un patrón de 102 píxeles como una matriz 6x17
    
    Args:
        features: Array de 102 elementos (0 o 1)
        label: Clase del patrón (0 o 1)
        ancho: Ancho de la matriz (default 17)
        alto: Alto de la matriz (default 6)
    """
    print(f"\n{'='*50}")
    print(f"CLASE: {int(label)}")
    print(f"{'='*50}")
    
    # Reshape a matriz 6x17
    matriz = features.reshape(alto, ancho)
    
    # Imprimir usando caracteres visuales
    for fila in matriz:
        linea = ""
        for pixel in fila:
            if pixel == 1:
                linea += "██"  # Pixel negro (activo)
            else:
                linea += "  "  # Pixel blanco (inactivo)
        print(linea)
    print()


def leer_y_visualizar_csv(filepath, n_patrones=5, mostrar_todos=False):
    """
    Lee un CSV y visualiza los primeros N patrones
    
    Args:
        filepath: Ruta al archivo CSV
        n_patrones: Número de patrones a mostrar
        mostrar_todos: Si True, muestra todos los patrones
    """
    print(f"\n{'#'*60}")
    print(f"ARCHIVO: {filepath}")
    print(f"{'#'*60}")
    
    # Leer CSV
    data = np.loadtxt(filepath, delimiter=';')
    
    # Separar features y labels
    X = data[:, :-1]  # Todas las columnas menos la última (102 features)
    y = data[:, -1]   # Última columna (label)
    
    print(f"\nTotal de patrones: {len(X)}")
    print(f"Dimensión features: {X.shape[1]}")
    print(f"Clases únicas: {np.unique(y)}")
    print(f"Distribución de clases:")
    for clase in np.unique(y):
        count = np.sum(y == clase)
        print(f"  Clase {int(clase)}: {count} ejemplos ({count/len(y)*100:.1f}%)")
    
    # Mostrar patrones
    if mostrar_todos:
        n = len(X)
    else:
        n = min(n_patrones, len(X))
    
    print(f"\n{'='*60}")
    print(f"MOSTRANDO {n} PATRONES:")
    print(f"{'='*60}")
    
    for i in range(n):
        print(f"\nPatrón #{i+1}")
        visualizar_patron(X[i], y[i])
        
        if not mostrar_todos and i < n-1:
            input("Presiona Enter para ver el siguiente patrón...")


def comparar_original_vs_distorsionado(original_path, distorsionado_path, indice=0):
    """
    Compara un patrón original con su versión distorsionada
    
    Args:
        original_path: Ruta al CSV original
        distorsionado_path: Ruta al CSV distorsionado
        indice: Índice del patrón a comparar
    """
    # Leer ambos archivos
    data_orig = np.loadtxt(original_path, delimiter=';')
    data_dist = np.loadtxt(distorsionado_path, delimiter=';')
    
    X_orig = data_orig[:, :-1]
    y_orig = data_orig[:, -1]
    
    X_dist = data_dist[:, :-1]
    y_dist = data_dist[:, -1]
    
    # Encontrar el patrón original en el distorsionado
    # (los primeros 10% son idénticos)
    n_originales = int(len(X_dist) * 0.1)
    
    print(f"\n{'#'*60}")
    print(f"COMPARACIÓN: ORIGINAL vs DISTORSIONADO")
    print(f"{'#'*60}")
    print(f"\nPatrones originales en el dataset distorsionado: {n_originales}")
    
    if indice < len(X_orig):
        print(f"\n--- PATRÓN ORIGINAL #{indice+1} ---")
        visualizar_patron(X_orig[indice], y_orig[indice])
        
        if indice < n_originales:
            print(f"\n--- MISMO PATRÓN EN DISTORSIONADO (SIN MODIFICAR) ---")
            visualizar_patron(X_dist[indice], y_dist[indice])
        
        # Buscar una versión distorsionada
        if indice + n_originales < len(X_dist):
            print(f"\n--- VERSIÓN DISTORSIONADA ---")
            visualizar_patron(X_dist[indice + n_originales], y_dist[indice + n_originales])
            
            # Calcular diferencias
            diff = np.abs(X_orig[indice] - X_dist[indice + n_originales])
            n_dif = np.sum(diff)
            porcentaje = (n_dif / len(X_orig[indice])) * 100
            print(f"Píxeles diferentes: {n_dif}/102 ({porcentaje:.1f}%)")


def menu_principal():
    """Menú interactivo para explorar los datasets"""
    
    # Rutas base
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    
    datasets = {
        '1': ('originales/100/letras.csv', 'Original 100 ejemplos'),
        '2': ('originales/500/letras.csv', 'Original 500 ejemplos'),
        '3': ('originales/1000/letras.csv', 'Original 1000 ejemplos'),
        '4': ('distorsionados/100/letras.csv', 'Distorsionado 100 ejemplos'),
        '5': ('distorsionados/500/letras.csv', 'Distorsionado 500 ejemplos'),
        '6': ('distorsionados/1000/letras.csv', 'Distorsionado 1000 ejemplos'),
    }
    
    while True:
        print(f"\n{'='*60}")
        print("VISUALIZADOR DE DATASETS")
        print(f"{'='*60}")
        print("\nOpciones:")
        print("1. Ver Original 100 ejemplos")
        print("2. Ver Original 500 ejemplos")
        print("3. Ver Original 1000 ejemplos")
        print("4. Ver Distorsionado 100 ejemplos")
        print("5. Ver Distorsionado 500 ejemplos")
        print("6. Ver Distorsionado 1000 ejemplos")
        print("7. Comparar Original vs Distorsionado (100)")
        print("8. Comparar Original vs Distorsionado (500)")
        print("9. Comparar Original vs Distorsionado (1000)")
        print("0. Salir")
        
        opcion = input("\nElige una opción: ").strip()
        
        if opcion == '0':
            print("¡Hasta luego!")
            break
        
        elif opcion in datasets:
            filepath = os.path.join(base_path, datasets[opcion][0])
            
            n = input("\n¿Cuántos patrones quieres ver? (Enter = 5, 'all' = todos): ").strip()
            
            if n.lower() == 'all':
                leer_y_visualizar_csv(filepath, mostrar_todos=True)
            else:
                try:
                    n = int(n) if n else 5
                    leer_y_visualizar_csv(filepath, n_patrones=n)
                except ValueError:
                    print("Número inválido, mostrando 5...")
                    leer_y_visualizar_csv(filepath, n_patrones=5)
        
        elif opcion == '7':
            orig = os.path.join(base_path, 'originales/100/letras.csv')
            dist = os.path.join(base_path, 'distorsionados/100/letras.csv')
            idx = input("¿Qué patrón quieres comparar? (0-99, Enter=0): ").strip()
            idx = int(idx) if idx else 0
            comparar_original_vs_distorsionado(orig, dist, idx)
        
        elif opcion == '8':
            orig = os.path.join(base_path, 'originales/500/letras.csv')
            dist = os.path.join(base_path, 'distorsionados/500/letras.csv')
            idx = input("¿Qué patrón quieres comparar? (0-499, Enter=0): ").strip()
            idx = int(idx) if idx else 0
            comparar_original_vs_distorsionado(orig, dist, idx)
        
        elif opcion == '9':
            orig = os.path.join(base_path, 'originales/1000/letras.csv')
            dist = os.path.join(base_path, 'distorsionados/1000/letras.csv')
            idx = input("¿Qué patrón quieres comparar? (0-999, Enter=0): ").strip()
            idx = int(idx) if idx else 0
            comparar_original_vs_distorsionado(orig, dist, idx)
        
        else:
            print("Opción inválida")


if __name__ == "__main__":
    # Si se pasa un argumento, visualizar ese archivo directamente
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        leer_y_visualizar_csv(filepath, n_patrones=n)
    else:
        # Menú interactivo
        menu_principal()
