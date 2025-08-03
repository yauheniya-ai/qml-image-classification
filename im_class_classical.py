import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
import pennylane as qml # needed just for loading the dataset

plt.style.use('dark_background') 

# Load dataset (plus-minus is 4-class)
ds_list = qml.data.load("plus-minus") # download dataset

#ds_list = qml.data.Dataset.open("datasets/plus-minus/plus-minus.h5") # open from a local file
ds = ds_list[0]

X_train, Y_train = ds.img_train, ds.labels_train  # labels in {0,1,2,3}
X_test, Y_test = ds.img_test, ds.labels_test

print("Train:", X_train.shape, Y_train.shape)
print("Test:", X_test.shape, Y_test.shape)

# Display 8 randomly selected images
fig, axes = plt.subplots(2, 4, figsize=(8, 4))  # 2 rows × 4 columns
axes = axes.flatten()

tick_positions = np.arange(0, 16, 2)  

# For each class (0–3), pick two random images
for class_label in range(4):
    # Get indices of samples belonging to the class
    class_indices = np.where(Y_train == class_label)[0]

    # Randomly choose 2 unique samples from that class
    chosen_indices = np.random.choice(class_indices, size=2, replace=False)

    for row in range(2):
        idx = chosen_indices[row]
        ax = axes[class_label + row * 4]  # Place: col = class_label, row = 0 or 1

        ax.imshow(X_train[idx], cmap='gray')
        ax.set_title(f"Label: {Y_train[idx]}", fontsize=8)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(axis='both', labelsize=6)

fig.suptitle("Plus Minus Dataset", fontsize=12)
plt.tight_layout()
plt.savefig("./output/dataset_plus_minus_8_images_by_class.png")
plt.close()

# Add channel dimension (for grayscale)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Convert labels to one-hot encoded vectors
num_classes = 4  # as per your dataset
Y_train_cat = to_categorical(Y_train, num_classes)
Y_test_cat = to_categorical(Y_test, num_classes)

# Build a simple CNN model
model = models.Sequential([
    Input(shape=X_train.shape[1:]),    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
print(model.summary())

# Record start time
start_time = time.time()

# Train the model
history = model.fit(X_train, Y_train_cat, epochs=30, batch_size=32, validation_split=0.2)

# Record end time and print training duration
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, Y_test_cat, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training accuracy and loss curves side-by-side

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot accuracy curve
axes[0].plot(history.history['accuracy'], label='Train Accuracy', c="#52bcff")
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', c="#94e047")
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Plot loss curve
axes[1].plot(history.history['loss'], label='Train Loss', c="#ea2081")
axes[1].plot(history.history['val_loss'], label='Validation Loss', c="#8b76e9")
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

fig.suptitle("Classical ML", fontsize=14)

plt.tight_layout()
plt.savefig("./output/classical_ml_training_accuracy_loss.png")  # Save the plot locally
plt.close()


# Prepare 2 random samples for each class (0 to 3)
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
axes = axes.flatten()

tick_positions = np.arange(0, 16, 2)

for class_label in range(4):
    # Get indices in test set where label == class_label
    class_indices = np.where(Y_test == class_label)[0]

    # Randomly select 2 samples from this class
    selected_indices = np.random.choice(class_indices, size=2, replace=False)

    for row in range(2):
        idx = selected_indices[row]
        img = X_test[idx]

        # Prepare image for display
        if img.shape[-1] == 1:
            img_disp = np.array(img).squeeze(-1)
        else:
            img_disp = img

        # Predict class
        pred_probs = model.predict(img[np.newaxis, ...], verbose=0)
        pred_label = np.argmax(pred_probs, axis=1)[0]
        true_label = Y_test[idx]

        # Determine subplot position
        ax = axes[class_label + row * 4]
        ax.imshow(img_disp, cmap='gray')
        ax.set_title(f"True: {true_label}, Pred: {pred_label}", fontsize=8)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions.astype(int))
        ax.set_yticklabels(tick_positions.astype(int))
        ax.tick_params(axis='both', labelsize=6)

fig.suptitle("Classical ML", fontsize=14)
plt.tight_layout()
plt.savefig("./output/classical_ml_test_images_pred_by_class.png")
plt.close()
