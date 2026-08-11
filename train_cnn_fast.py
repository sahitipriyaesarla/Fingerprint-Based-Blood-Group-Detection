# train_cnn_fast.py - Fast CNN training for high accuracy
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

# ---------------- CONFIG ----------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 64  # Large batch for faster training
CLASSES = 8
EPOCHS = 30  # Fast but effective

# Dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

print("🚀 Starting Fast CNN Training...")
print(f"📊 Dataset: {train_dir}")
print(f"🎯 Classes: {CLASSES}")
print(f"⚡ Batch Size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")
print("-" * 50)

# ---------------- DATA GENERATORS ----------------
# Enhanced data augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

# Display class indices
print("\n📋 CLASS INDICES:")
for class_name, index in train_data.class_indices.items():
    print(f"  {class_name}: {index}")
print("-" * 50)

# ---------------- CNN MODEL ----------------
model = Sequential([
    Input(shape=(128, 128, 1)),
    
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.35),
    
    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.4),
    
    # Block 4
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.45),
    
    # Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(CLASSES, activation='softmax')
])

# Compile with optimized learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🧠 Model Architecture:")
model.summary()
print("-" * 50)

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)

# ---------------- TRAINING ----------------
print("\n🏃‍♂️ Starting Training...")
start_time = time.time()

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

end_time = time.time()
training_time = end_time - start_time

print(f"\n⏱️ Training completed in {training_time/60:.1f} minutes")

# ---------------- RESULTS ----------------
# Get final metrics
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
best_val_acc = max(history.history['val_accuracy'])

print("\n📊 TRAINING RESULTS:")
print(f"  Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"  Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"  Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# Evaluate on test set
print("\n🧪 Evaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ---------------- SAVE MODEL ----------------
model.save("utils/models/blood_group_model.h5")
print("\n✅ Model saved successfully to: utils/models/blood_group_model.h5")

print("\n🎯 TRAINING SUMMARY:")
print(f"  ⚡ Training Time: {training_time/60:.1f} minutes")
print(f"  📈 Best Accuracy: {best_val_acc*100:.2f}%")
print(f"  💾 Model Saved: blood_group_model.h5")
print(f"  🎯 Ready for Prediction!")
