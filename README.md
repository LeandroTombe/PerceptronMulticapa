# Perceptrón Multicapa (MLP) - TP 2025

Implementación desde cero de un Perceptrón Multicapa para clasificación de patrones.

## 📁 Estructura del Proyecto

```
perceptron/
├── src/                        # Código fuente principal
│   ├── ActivationFunctions.py  # Funciones de activación (ReLU, Sigmoid, Tanh, Softmax)
│   ├── DataLoader.py           # Cargador de datasets (MNIST y custom)
│   ├── Distorsionador.py       # Generador de datasets con distorsión
│   └── __init__.py
│
├── tests/                      # Tests unitarios
│   ├── test_activation_functions.py
│   ├── test_distorsionador.py
│   ├── test_dataloader.py
│   └── __init__.py
│
├── utils/                      # Utilidades y scripts de análisis
│   ├── analizar_datasets.py    # Análisis de datasets
│   ├── verificar_distorsion.py # Verificación de distorsión
│   └── __init__.py
│
├── scripts/                    # Scripts ejecutables
│   └── test_basics.py          # Test básico de componentes
│
├── datasets/                   # Datasets del proyecto
│   ├── originales/             # Patrones base sin distorsión
│   │   ├── 100/letras.csv
│   │   ├── 500/letras.csv
│   │   └── 1000/letras.csv
│   └── distorsionados/         # Datasets con 10% original + 90% distorsionado
│       ├── 100/letras.csv
│       ├── 500/letras.csv
│       └── 1000/letras.csv
│
├── .venv/                      # Entorno virtual Python
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo
```

---

## 🚀 Instalación

### 1. Crear y activar entorno virtual

**Windows PowerShell:**
```powershell
# Crear venv
python -m venv .venv

# Activar (puede requerir permisos)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🧪 Tests

### Ejecutar todos los tests
```bash
python -m pytest tests/ -v
```

### Ejecutar tests individuales
```bash
python -m pytest tests/test_activation_functions.py -v
python -m pytest tests/test_distorsionador.py -v
python -m pytest tests/test_dataloader.py -v
```

### Ejecutar script de test básico
```bash
python scripts/test_basics.py
```

---

## 📊 Datasets

### Características de los Datasets

- **Formato**: CSV con separador `;`
- **Estructura**: 102 features (píxeles) + 1 label (clase)
- **Valores**: Binarios (0 o 1)
- **Clases**: 2 (distribución 2:1 aproximadamente)

### Datasets Originales
Contienen 3 patrones base únicos repetidos:
- `datasets/originales/100/letras.csv` (100 ejemplos)
- `datasets/originales/500/letras.csv` (500 ejemplos)
- `datasets/originales/1000/letras.csv` (1000 ejemplos)

### Datasets Distorsionados
Cumplen requisitos del TP (10% sin distorsión + 90% con distorsión 1-30%):
- `datasets/distorsionados/100/letras.csv` (10 originales + 90 distorsionados)
- `datasets/distorsionados/500/letras.csv` (50 originales + 450 distorsionados)
- `datasets/distorsionados/1000/letras.csv` (100 originales + 900 distorsionados)

### Generar nuevos datasets distorsionados
```bash
python -m src.Distorsionador
```

### Analizar datasets
```bash
python utils/analizar_datasets.py
python utils/verificar_distorsion.py
```

---

## 📦 Componentes Implementados

### ✅ ActivationFunctions
- `relu(z)` y `relu_derivative(z)`
- `sigmoid(z)` y `sigmoid_derivative(z)`
- `tanh(z)` y `tanh_derivative(z)`
- `softmax(z)` para capa de salida

### ✅ Distorsionador
- Carga de patrones base
- Generación de datasets con distorsión controlada (1-30%)
- Distribución configurables de clases
- Exportación a CSV

### ✅ DataLoader
- Carga de MNIST usando sklearn
- Preprocesamiento (normalización)
- One-hot encoding
- Train/test split

---

## 🔧 Uso

### Ejemplo: Generar datasets
```python
from src.Distorsionador import Distorsionador

# Crear generador
dist = Distorsionador(seed=42)

# Cargar patrones base
dist.cargar_patrones_base("datasets/originales/100/letras.csv")

# Generar dataset
X, y = dist.generar_dataset(
    n_ejemplos=100,
    output_path="datasets/distorsionados/100/letras.csv",
    distribucion_clases={0: 0.67, 1: 0.33}
)
```

### Ejemplo: Usar funciones de activación
```python
from src.ActivationFunctions import ActivationFunction
import numpy as np

x = np.array([[1.0, -2.0, 3.0]])

# Aplicar activaciones
relu_out = ActivationFunction.relu(x)
sigmoid_out = ActivationFunction.sigmoid(x)
tanh_out = ActivationFunction.tanh(x)

# Derivadas para backpropagation
relu_grad = ActivationFunction.relu_derivative(x)
```

---

## 📝 Requisitos del TP

- [x] Generar 3 datasets: 100, 500, 1000 ejemplos
- [x] 10% patrones sin distorsionar
- [x] 90% patrones con distorsión del 1% al 30%
- [x] Distribución representativa de clases
- [ ] Implementar MLP con 2-3 capas ocultas
- [ ] Funciones de activación: ReLU, Sigmoid, Tanh
- [ ] Optimizadores: SGD, Momentum, Adam, RMSProp
- [ ] Función de pérdida: Cross-Entropy
- [ ] Forward y Backward propagation
- [ ] Entrenamiento y evaluación
- [ ] Matriz de confusión
- [ ] Gráficos de pérdida y accuracy

---

## 🎯 Próximos Pasos

1. **Implementar clase `Perceptron`** (neurona individual)
2. **Implementar clase `MLP`** (red multicapa completa)
3. **Implementar optimizadores** (SGD, Momentum, Adam, RMSProp)
4. **Implementar función de pérdida** (Cross-Entropy)
5. **Crear módulo de métricas** (accuracy, confusion matrix)
6. **Crear módulo de visualización** (plots de training)
7. **Ejecutar experimentos** con diferentes configuraciones
8. **Generar informe** con resultados y conclusiones

---

## 📚 Dependencias

- `numpy>=1.21.0` - Operaciones matriciales
- `scikit-learn>=1.0.0` - Datasets y métricas
- `matplotlib>=3.5.0` - Visualización
- `seaborn>=0.11.0` - Gráficos estadísticos
- `tqdm>=4.62.0` - Barras de progreso

---

## 👨‍💻 Autor

Trabajo Práctico - Perceptrón Multicapa 2025

---

## 📄 Licencia

Proyecto académico - TP 2025
