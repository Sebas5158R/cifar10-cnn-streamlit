import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración básica de la página web
st.set_page_config(page_title="Clasificador CIFAR-10", layout="centered")

st.title("🛸 Clasificador de Imágenes con CNN")
st.write("Sube una imagen para que la Inteligencia Artificial prediga qué objeto es (Entrenado con CIFAR-10).")

# =========================================================================
# SECCIÓN NUEVA: Información de Clases Soportadas (Mejora de UX)
# =========================================================================
with st.expander("📌 Ver las 10 categorías que este modelo puede reconocer", expanded=True):
    st.write("Para obtener los mejores resultados, asegúrate de que tu imagen pertenezca a una de estas clases:")
    
    # Creamos dos columnas para organizar las etiquetas de manera limpia
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- ✈️ **Avión**")
        st.markdown("- 🚗 **Automóvil**")
        st.markdown("- 🐦 **Pájaro**")
        st.markdown("- 🐱 **Gato**")
        st.markdown("- 🦌 **Ciervo**")
    with col2:
        st.markdown("- 🐶 **Perro**")
        st.markdown("- 🐸 **Rana**")
        st.markdown("- 🐴 **Caballo**")
        st.markdown("- 🚢 **Barco**")
        st.markdown("- 🚚 **Camión**")
        
    st.caption("💡 *Nota: El modelo procesa las imágenes en formato 32x32 píxeles, por lo que funciona mejor con objetos individuales y centrados.*")

# =========================================================================

# Lista de clases (debe mantener exactamente el mismo orden que el entrenamiento)
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Cargar el modelo guardado utilizando caché para no recargarlo en cada clic
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('modelo_cifar10.keras')

try:
    model = load_my_model()
    st.success("🤖 ¡Modelo cargado correctamente!")
except Exception as e:
    st.error("No se encontró el archivo 'modelo_cifar10.keras'. Por favor ejecuta primero el entrenamiento.")
    st.stop()

# Componente para subir archivos web
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Mostrar la imagen original subida por el usuarioñ
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Subida', width=250)
    
    # 2. Preprocesar la imagen para que coincida con lo que espera el modelo (32x32x3 escalado)
    # Convertimos a RGB en caso de que suban una imagen PNG con canal alpha (transparencia)
    img_resized = image.convert('RGB').resize((32, 32))
    img_array = np.array(img_resized)
    img_array = img_array.astype('float32') / 255.0  # Normalización [0, 1]
    img_array = np.expand_dims(img_array, axis=0)     # Añadir dimensión de lote (Batch size = 1)
    
    # Botón para realizar la predicción
    if st.button('Clasificar Imagen'):
        with st.spinner('Analizando imagen...'):
            # Realizar predicción
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Asegurar formato de probabilidad distribución
            
            # Obtener el índice con mayor probabilidad
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            # 3. Mostrar resultados en la app web
            st.subheader("Resultados:")
            st.metric(label="Clase Predicha", value=predicted_class)
            st.metric(label="Nivel de Confianza / Probabilidad", value=f"{confidence:.2f}%")
            
            # Opcional: Mostrar gráfico de barras con el porcentaje de todas las clases
            st.write("### Probabilidades detalladas:")
            chart_data = {class_names[i]: float(predictions[0][i]) for i in range(10)}
            st.bar_chart(chart_data)
