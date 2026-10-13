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

### 2. Instalar dependencias ligeras
Dado que el modelo ya está entrenado, no necesitas descargar paquetes pesados de GPU en tu PC. Abre tu terminal en la carpeta del proyecto e instala los requerimientos básicos:
```bash
pip install -r requirements.txt
```

### 3. Lanzar la Aplicación Web
Para levantar el servidor local de Streamlit y probar el clasificador con tus propias imágenes, ejecuta:
```bash
streamlit run app.py
```
Automáticamente se abrirá una pestaña en tu navegador web en la dirección `http://localhost:8501`. ¡Arrastra cualquier imagen y observa la predicción en tiempo real!
