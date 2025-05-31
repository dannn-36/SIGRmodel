# -*- coding: utf-8 -*-
"""Clasificación de imágenes optimizada para usar más RAM"""

import os
import zipfile
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import mixed_precision

import warnings
warnings.filterwarnings('ignore')

#  Habilitar Mixed Precision para reducir consumo de VRAM
mixed_precision.set_global_policy('mixed_float16')

#  Configurar la GPU para permitir crecimiento dinámico de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(" GPU configurada con crecimiento de memoria dinámico")
    except RuntimeError as e:
        print(e)

#  Configuración del dataset
data_dir = pathlib.Path("/home/daniel/Documents/SIGR/SIRG DATASET/train")
if not data_dir.exists():
    raise FileNotFoundError(f"X ERROR: El directorio no existe: {data_dir}")

#  Parámetros de imágenes
batch_size = 20  #  Aumentado para usar más RAM
img_height = 400
img_width = 400
AUTOTUNE = tf.data.AUTOTUNE

#  Cargar datasets con más uso de RAM
print(" Cargando TRAINING DATASET...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("\n Cargando TESTING DATASET...")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#  Obtener clases
class_names = train_ds.class_names
print("\n CLASES:", class_names)

#  Configurar dataset para rendimiento **(más uso de RAM)**
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#  Augmentación de datos
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

#  Modelo optimizado
num_classes = len(class_names)
model = Sequential([
    data_augmentation,
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),  # Elimina un 30% de neuronas aleatoriamente
    layers.Dense(num_classes),

])

#  Compilar modelo
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.build(input_shape=(None, img_height, img_width, 3))
model.summary()


num_batches = tf.data.experimental.cardinality(train_ds).numpy()
print(f" Número total de batches en train_ds: {num_batches}")

for image_batch, label_batch in train_ds.take(1):  # Tomar solo 1 batch
    print(f" Tamaño del batch: {image_batch.shape}") 

#  Entrenamiento del modelo
epochs = 40
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

#  Guardar el modelo
model.save('/home/daniel/Documents/SIGR/model_ver1.1.keras')

#  Convertir modelo a TensorFlow Lite (optimizado para float16)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]  #  Habilita TF Select
tflite_model = converter.convert()

#  Guardar modelo en formato .tflite
tflite_path = "/home/daniel/Documents/SIGR/model_ver1.1.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f" Modelo guardado en: {tflite_path}")

#  Función para predicción de imágenes
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"X No se encontró la imagen: {image_path}")
        return

    img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)

    # Normalizar la imagen (como en el entrenamiento)
    img_array = img_array / 255.0  # Normalizar a rango [0,1]

    img_array = tf.expand_dims(img_array, 0)  # Crear batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(f"🔍 La imagen es probablemente {class_names[np.argmax(score)]} con una confianza de {100 * np.max(score):.2f}%.")

#  Prueba de predicción con una imagen
test_image_path = "/home/daniel/Documents/SIGR/Images/test/camisa.jfif"
predict_image(test_image_path)

#  Visualizar los resultados de entrenamiento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
