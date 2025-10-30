"""Generador de datasets con distorsión para el TP de Perceptrón Multicapa.

Requisitos del TP:
- 10% de ejemplos sin distorsionar (patrones originales)
- 90% de ejemplos con distorsión del 1% al 30%
- Distribución representativa de clases
"""
import numpy as np
import os


class Distorsionador:
    """Clase para generar datasets con distorsión a partir de patrones base."""
    
    def __init__(self, seed=42):
        """
        Args:
            seed: Semilla para reproducibilidad de resultados
        """
        self.seed = seed
        np.random.seed(seed)
        self.patrones_base = []
        
    def cargar_patrones_base(self, filepath):
        """Carga los patrones base desde un archivo CSV.
        
        Args:
            filepath: Ruta al archivo CSV con patrones originales
            
        Returns:
            Array con patrones únicos (sin labels si las tiene)
        """
        data = np.loadtxt(filepath, delimiter=';', dtype=int)
        
        # Separar features y labels si existen
        if data.shape[1] == 103:  # 102 features + 1 label
            features = data[:, :-1]
            labels = data[:, -1]
            
            # Extraer patrones únicos con sus labels
            unique_patterns = []
            for label in np.unique(labels):
                # Tomar el primer patrón de cada clase
                pattern_idx = np.where(labels == label)[0][0]
                pattern = features[pattern_idx]
                unique_patterns.append((pattern, label))
            
            self.patrones_base = unique_patterns
            print(f"✓ Cargados {len(unique_patterns)} patrones base con labels")
            
        else:  # Solo features
            unique_features = np.unique(data, axis=0)
            self.patrones_base = [(p, 0) for p in unique_features]
            print(f"✓ Cargados {len(unique_features)} patrones base sin labels")
        
        return self.patrones_base
    
    def distorsionar_patron(self, patron, porcentaje_distorsion):
        """Aplica distorsión a un patrón binario.
        
        Args:
            patron: Array de 0s y 1s (shape: (102,))
            porcentaje_distorsion: Porcentaje de píxeles a modificar (1-30)
            
        Returns:
            Patrón distorsionado
        """
        patron_dist = patron.copy()
        n_pixeles = len(patron)
        
        # Calcular cantidad de píxeles a modificar
        n_cambios = int(np.ceil(n_pixeles * porcentaje_distorsion / 100.0))
        
        # Seleccionar píxeles aleatorios para modificar
        indices = np.random.choice(n_pixeles, size=n_cambios, replace=False)
        
        # Invertir los píxeles seleccionados (0->1, 1->0)
        patron_dist[indices] = 1 - patron_dist[indices]
        
        return patron_dist
    
    def generar_dataset(self, n_ejemplos, output_path=None, distribucion_clases=None):
        """Genera un dataset completo con distorsión.
        
        Args:
            n_ejemplos: Cantidad total de ejemplos a generar (100, 500, 1000)
            output_path: Ruta donde guardar el dataset (opcional)
            distribucion_clases: Dict con proporción de cada clase, ej: {0: 0.67, 1: 0.33}
            
        Returns:
            Tuple (X, y) con features y labels
        """
        if not self.patrones_base:
            raise ValueError("Primero debes cargar los patrones base con cargar_patrones_base()")
        
        # Calcular cantidad de ejemplos sin distorsión (10%)
        n_sin_distorsion = int(np.ceil(n_ejemplos * 0.10))
        n_con_distorsion = n_ejemplos - n_sin_distorsion
        
        print(f"\nGenerando dataset de {n_ejemplos} ejemplos:")
        print(f"  - {n_sin_distorsion} sin distorsión (10%)")
        print(f"  - {n_con_distorsion} con distorsión (90%)")
        
        # Determinar distribución de clases
        if distribucion_clases is None:
            # Por defecto: distribuir equitativamente entre clases disponibles
            n_clases = len(self.patrones_base)
            distribucion_clases = {i: 1.0/n_clases for i in range(n_clases)}
        
        X_list = []
        y_list = []
        
        # 1. Generar ejemplos SIN distorsión (10%)
        for i in range(n_sin_distorsion):
            # Seleccionar clase según distribución
            clase_idx = np.random.choice(
                list(distribucion_clases.keys()), 
                p=list(distribucion_clases.values())
            )
            patron, label = self.patrones_base[clase_idx]
            
            X_list.append(patron.copy())
            y_list.append(label)
        
        # 2. Generar ejemplos CON distorsión (90%)
        for i in range(n_con_distorsion):
            # Seleccionar clase según distribución
            clase_idx = np.random.choice(
                list(distribucion_clases.keys()), 
                p=list(distribucion_clases.values())
            )
            patron, label = self.patrones_base[clase_idx]
            
            # Aplicar distorsión aleatoria entre 1% y 30%
            porcentaje = np.random.uniform(1.0, 30.0)
            patron_distorsionado = self.distorsionar_patron(patron, porcentaje)
            
            X_list.append(patron_distorsionado)
            y_list.append(label)
        
        # Convertir a arrays numpy
        X = np.array(X_list, dtype=int)
        y = np.array(y_list, dtype=int)
        
        # Mezclar ejemplos
        indices = np.random.permutation(n_ejemplos)
        X = X[indices]
        y = y[indices]
        
        print(f"✓ Dataset generado: {X.shape}")
        print(f"  Distribución de clases: {np.bincount(y)}")
        
        # Guardar si se especifica ruta
        if output_path:
            self._guardar_dataset(X, y, output_path)
        
        return X, y
    
    def _guardar_dataset(self, X, y, output_path):
        """Guarda el dataset en formato CSV (features + label).
        
        Args:
            X: Features (n_ejemplos, 102)
            y: Labels (n_ejemplos,)
            output_path: Ruta completa del archivo de salida
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Combinar features y labels
        data = np.column_stack([X, y])
        
        # Guardar en formato CSV
        np.savetxt(output_path, data, delimiter=';', fmt='%d')
        
        print(f"✓ Dataset guardado en: {output_path}")


def main():
    """Ejemplo de uso del Distorsionador."""
    
    print("="*70)
    print("GENERADOR DE DATASETS CON DISTORSIÓN")
    print("="*70)
    
    # Crear instancia del distorsionador
    dist = Distorsionador(seed=42)
    
    # Cargar patrones base desde el dataset original de 100
    base_path = "datasets/originales/100/letras.csv"
    dist.cargar_patrones_base(base_path)
    
    # Distribución 2:1 (clase 0: 67%, clase 1: 33%)
    distribucion = {0: 0.67, 1: 0.33}
    
    # Generar los 3 datasets
    datasets_config = [
        (100, "datasets/distorsionados/100/letras.csv"),
        (500, "datasets/distorsionados/500/letras.csv"),
        (1000, "datasets/distorsionados/1000/letras.csv")
    ]
    
    for n_ejemplos, output_path in datasets_config:
        X, y = dist.generar_dataset(
            n_ejemplos=n_ejemplos,
            output_path=output_path,
            distribucion_clases=distribucion
        )
    
    print("\n" + "="*70)
    print("✓ GENERACIÓN COMPLETADA")
    print("="*70)
    print("\nDatasets generados en: datasets/distorsionados/")
    print("  - 100/letras.csv  (100 ejemplos)")
    print("  - 500/letras.csv  (500 ejemplos)")
    print("  - 1000/letras.csv (1000 ejemplos)")


if __name__ == "__main__":
    main()
