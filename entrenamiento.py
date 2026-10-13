import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, mixed_precision
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================================
# 0. CONFIGURACIÓN DE VELOCIDAD (GPU)
# ==========================================
# Mantener la velocidad de la GPU mediante precisión mixta
mixed_precision.set_global_policy('mixed_float16')

# ==========================================
# 1. SELECCIÓN Y DESCRIPCIÓN DEL DATASET
# ==========================================
print("Cargando el dataset CIFAR-10...")
(X_train_full, y_train_full), (X_test_full, y_test_full) = datasets.cifar10.load_data()

class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# ==========================================
# 2. EXPLORACIÓN Y VISUALIZACIÓN
# ==========================================
print("\nMostrando una muestra de imágenes...")
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train_full[i])
    plt.xlabel(class_names[y_train_full[i][0]])
plt.tight_layout()
plt.show()

# ==========================================
# 3 & 4. PREPROCESAMIENTO, NORMALIZACIÓN Y DIVISIÓN
# ==========================================
# Normalizar los datos a rango [0, 1]
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test_full.astype('float32') / 255.0
y_test = y_test_full

# Separar 5,000 para validación y usar 45,000 para entrenar
X_val = X_train_full[-5000:]
y_val = y_train_full[-5000:]
X_train = X_train_full[:-5000]
y_train = y_train_full[:-5000]

print(f"\nConjunto de Entrenamiento: {X_train.shape} imágenes")
print(f"Conjunto de Validación: {X_val.shape} imágenes")
print(f"Conjunto de Pruebas: {X_test.shape} imágenes")

# ==========================================
# 5 & 6. CONSTRUCCIÓN DE LA CNN + DATA AUGMENTATION
# ==========================================
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    data_augmentation, 
    
    # Bloque Convolucional 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25), 
    
    # Bloque Convolucional 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Clasificador Densa
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax', dtype='float32') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ==========================================
# ENTRENAMIENTO DEL MODELO (Fijado con History)
# ==========================================
print("\nIniciando entrenamiento de Alta Precisión...")
history = model.fit(X_train, y_train, 
                    epochs=20, 
                    validation_data=(X_val, y_val), 
                    batch_size=256)

# ==========================================
# 7 & 8. EVALUACIÓN Y GRÁFICAS DEL ENTRENAMIENTO
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nPrecisión exacta en el set de prueba (Accuracy): {test_acc:.4f}")

# Gráfica de curvas de pérdida y precisión
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Precisión del Modelo')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Pérdida del Modelo')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# ==========================================
# 9. MATRIZ DE CONFUSIÓN
# ==========================================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ==========================================
# 10. MÍNIMO 10 PREDICCIONES CON IMÁGENES
# ==========================================
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i])
    
    true_label = class_names[y_test[i][0]]
    pred_label = class_names[y_pred[i]]
    confidence = y_pred_probs[i][y_pred[i]] * 100
    
    color = 'green' if true_label == pred_label else 'red'
    plt.xlabel(f"Pred: {pred_label}\n({confidence:.1f}%)", color=color)
plt.tight_layout()
plt.show()

# ==========================================
# 11. GUARDADO DEL MODELO ENTRENADO
# ==========================================
model_path = 'modelo_cifar10.keras'
model.save(model_path)
print(f"\n¡Modelo guardado exitosamente en '{model_path}'!")
