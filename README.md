# 🛸 Clasificador de Imágenes CNN con Streamlit (CIFAR-10)

Este proyecto consiste en una solución de Deep Learning de extremo a extremo que utiliza una **Red Neuronal Convolucional (CNN)** entrenada con el dataset CIFAR-10 para clasificar imágenes en 10 categorías distintas. 

Para optimizar los recursos de cómputo, el flujo de trabajo se dividió en un enfoque híbrido: la experimentación y el entrenamiento pesado se realizaron en la nube mediante **Google Colab (aprovechando una GPU T4)**, mientras que el modelo resultante se exportó para ser ejecutado localmente a través de una interfaz web interactiva desarrollada en **Streamlit**.

## 📌 Categorías Soportadas
El modelo es capaz de reconocer las siguientes 10 clases:
* Avión, Automóvil, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión.

## 📁 Estructura del Repositorio

* `notebook_experimento.ipynb`: Cuaderno de **Google Colab** utilizado para la carga del dataset, visualización, diseño de la arquitectura CNN, entrenamiento optimizado con precisión mixta (Mixed Precision) y exportación del modelo.
* `entrenamiento.py`: Script de respaldo en Python que replica la lógica y arquitectura de bloques convolucionales ejecutada en la nube.
* `modelo_cifar10.keras`: Archivo binario con los pesos del modelo entrenado en Colab, listo para producción.
* `app.py`: Aplicación web interactiva de **Streamlit** que carga el modelo localmente y procesa las imágenes del usuario.
* `requirements.txt`: Archivo de dependencias optimizado para la ejecución local (utiliza versiones CPU más ligeras).

## 🧠 Flujo de Entrenamiento (Google Colab)
El modelo fue desarrollado en Google Colab para garantizar velocidad y precisión: https://colab.research.google.com/drive/1zau5xBAG5xJWq0NV4dh04JDPpG5evKJ8?usp=sharing
1. **Infraestructura:** Conexión a un entorno de ejecución con aceleración por **GPU T4**.
2. **Optimización:** Implementación de una política de **precisión mixta (`mixed_float16`)**, permitiendo un entrenamiento ultra veloz (menos de un minuto) con un tamaño de lote (*batch size*) de 512.
3. **Métricas visuales:** El cuaderno incluye el histórico de precisión/pérdida, la matriz de confusión y el reporte de clasificación por clase antes de exportar el archivo `.keras`.

## 🚀 Instrucciones de Ejecución Local

### 1. Prerrequisitos
Asegúrate de tener Python instalado en tu sistema (versión 3.9 a 3.11 recomendada).

### 2. Clonar el repositorio y Crear el Entorno Virtual
Abre tu terminal (CMD, PowerShell o Git Bash), navega hasta la carpeta del proyecto y ejecuta los siguientes comandos según tu sistema operativo:

**En Windows (CMD o PowerShell):**
```bash
# Crear el entorno virtual llamado 'venv'
python -m venv venv

# Activar el entorno virtual
.\venv\Scripts\activate
```

**En macOS / Linux (Terminal):**
```bash
# Crear el entorno virtual llamado 'venv'
python3 -m venv venv

# Activar el entorno virtual
source venv/bin/activate
```

*💡 Sabrás que se activó correctamente porque verás `(venv)` al inicio de la línea de tu terminal.*

### 3. Instalar dependencias ligeras
Con el entorno virtual activo, instala los requerimientos básicos del proyecto sin alterar el resto de tu sistema:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Lanzar la Aplicación Web
Para levantar el servidor local de Streamlit y probar el clasificador con tus propias imágenes, ejecuta:
```bash
streamlit run app.py
```
Automáticamente se abrirá una pestaña en tu navegador web en la dirección `http://localhost:8501`. ¡Arrastra cualquier imagen y observa la predicción en tiempo real!

---
*Nota: Cuando termines de usar la aplicación, puedes cerrar el entorno virtual en tu terminal escribiendo simplemente el comando `deactivate`.*

