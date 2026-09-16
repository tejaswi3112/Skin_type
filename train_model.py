import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

# ---------------- SETTINGS ----------------
DATASET_PATH = "dataset/"
img_size = 224
batch_size = 8
epochs = 15

# ---------------- DATA ----------------
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# ---------------- CHECK ----------------
if train_data.samples == 0:
    raise ValueError("❌ No images found in dataset folder!")

print("Classes:", train_data.class_indices)
print("Total images:", train_data.samples)

# ---------------- MODEL ----------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(len(train_data.class_indices), activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(train_data, epochs=epochs)

# ---------------- SAVE ----------------
model.save("skin_model.keras")

with open("classes.json", "w") as f:
    json.dump(list(train_data.class_indices.keys()), f)

print("✅ Training completed successfully!")