import kagglehub
import os
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras import layers, models

# -----------------------------
# Step 1: Data Loading
# -----------------------------
data_dir = kagglehub.dataset_download("alessiocorrado99/animals10")  # Update path if needed
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    directory=data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    directory=data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_data.num_classes
input_shape = img_size + (3,)

# -----------------------------
# Step 2: Define ZFNet Model
# -----------------------------
def build_zfnet():
    model = models.Sequential([
        layers.Conv2D(96, (7, 7), strides=2, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# -----------------------------
# Step 3: Define VGG16 Model
# -----------------------------
def build_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze the convolutional base

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# -----------------------------
# Step 4: Define GoogLeNet Model (InceptionV3 substitute)
# -----------------------------
def build_googlenet():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze base layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# -----------------------------
# Step 5: Training Function
# -----------------------------
def train_and_evaluate(model, name):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"\nTraining {name}...\n")
    start = time.time()
    history = model.fit(train_data, validation_data=val_data, epochs=10)
    end = time.time()
    val_loss, val_acc = model.evaluate(val_data, verbose=0)
    print(f"{name} Accuracy: {val_acc:.4f} | Time: {end - start:.2f} sec")
    return history, val_acc, end - start

# -----------------------------
# Step 6: Plot History
# -----------------------------
def plot_history(history, model_name):
    plt.figure(figsize=(10, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# -----------------------------
# Step 7: Run All Models
# -----------------------------
results = []

# ZFNet
zfnet = build_zfnet()
zfnet_history, zfnet_acc, zfnet_time = train_and_evaluate(zfnet, "ZFNet")
plot_history(zfnet_history, "ZFNet")
results.append(("ZFNet", zfnet_acc, zfnet_time))

# VGG16
vgg16 = build_vgg16()
vgg16_history, vgg16_acc, vgg16_time = train_and_evaluate(vgg16, "VGG16")
plot_history(vgg16_history, "VGG16")
results.append(("VGG16", vgg16_acc, vgg16_time))

# GoogLeNet (InceptionV3)
googlenet = build_googlenet()
googlenet_history, googlenet_acc, googlenet_time = train_and_evaluate(googlenet, "GoogLeNet (InceptionV3)")
plot_history(googlenet_history, "GoogLeNet (InceptionV3)")
results.append(("GoogLeNet", googlenet_acc, googlenet_time))

# -----------------------------
# Step 8: Print Model Comparison
# -----------------------------
print("\nModel Comparison:")
for model_name, acc, t in results:
    print(f"{model_name}: Accuracy = {acc:.4f}, Time = {t:.2f} sec")
