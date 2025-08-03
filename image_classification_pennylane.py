import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('dark_background') 

# Load dataset (plus-minus is 4-class)
ds = qml.data.load("plus-minus")[0]
X_train, Y_train = ds.img_train, ds.labels_train
X_test, Y_test = ds.img_test, ds.labels_test

num_classes = 4
def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

Y_train_oh = one_hot(Y_train, num_classes)
Y_test_oh = one_hot(Y_test, num_classes)

# Quantum device
dev = qml.device('default.qubit', wires=8)

@qml.qnode(dev, interface='autograd')
def circuit(image, params):
    qml.AmplitudeEmbedding(image.flatten(), normalize=True, wires=range(8))
    for i in range(4):
        for j in range(8):
            qml.RY(params[i, j], wires=j)
        for j in range(7):
            qml.CNOT(wires=[j, j + 1])
    return [qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))]

def model(image, params, weights, biases):
    expvals = pnp.array(circuit(image, params))     # Quantum processing
    logits = pnp.dot(weights, expvals) + biases     # Classical post-processing
    return logits

def cross_entropy(labels, logits):
    exps = pnp.exp(logits - pnp.max(logits))
    softmax = exps / pnp.sum(exps)
    return -pnp.sum(labels * pnp.log(softmax + 1e-10))

def batch_loss(params, weights, biases):
    loss = 0
    for x, y in zip(X_train, Y_train_oh):
        logits = model(x, params, weights, biases)
        loss += cross_entropy(y, logits)
    return loss / len(X_train)

def accuracy(params, weights, biases, X, Y):
    preds = []
    for x in X:
        logits = model(x, params, weights, biases)
        preds.append(np.argmax(logits))
    return np.mean(preds == Y)

# Initialize parameters
np.random.seed(0)
params = pnp.random.normal(0, np.pi, size=(4, 8), requires_grad=True)
weights = pnp.random.normal(0, 0.1, size=(num_classes, 2), requires_grad=True)
biases = pnp.zeros(num_classes, requires_grad=True)

opt = qml.optimize.NesterovMomentumOptimizer(stepsize=0.1)
steps = 200     # Adjust the number of steps 

cost_history = []
acc_history = []
step_times = []

print(f"Training for {steps} steps...")

start_time = time.time()
for i in range(steps):
    step_start = time.time()

    def closure(params_, weights_, biases_):
        return batch_loss(params_, weights_, biases_)

    (params, weights, biases), cost = opt.step_and_cost(closure, params, weights, biases)

    train_acc = accuracy(params, weights, biases, X_train, Y_train)
    step_time = time.time() - step_start
    elapsed = time.time() - start_time

    cost_history.append(cost)
    acc_history.append(train_acc)
    step_times.append(step_time)

    print(f"Step {i+1:2d}/{steps}, Cost: {cost:.4f}, Train acc: {train_acc:.4f}, "
          f"Step time: {step_time/60:.2f}min, Total elapsed: {elapsed/60:.2f}min")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTraining completed in {total_time:.2f} seconds.")
print(f"Average time per step: {np.mean(step_times):.2f} seconds.")

test_acc = accuracy(params, weights, biases, X_test, Y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save training history
history_df = pd.DataFrame({
    'Step': np.arange(1, steps + 1),
    'Train Accuracy': acc_history,
    'Train Loss': cost_history,
    'Step Time (s)': step_times
})
history_df.to_csv('./output/pennylane_qml_training_history.csv', index=False)

summary_df = pd.DataFrame({
    'Total Training Time (s)': [total_time],
    'Average Step Time (s)': [np.mean(step_times)],
    'Test Accuracy': [test_acc]
})
summary_df.to_csv('./output/pennylane_qml_training_summary.csv', index=False)

######## Visualizations ########

# --------------------------
# Plot Accuracy & Loss
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Accuracy
axes[0].plot(acc_history, label='Train Accuracy', c="#52bcff")
axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Loss
axes[1].plot(cost_history, label='Train Loss', c="#ea2081")
axes[1].set_title('Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

fig.suptitle("PennyLane QML", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("./output/pennylane_qml_training_accuracy_loss.png")
plt.close()

# --------------------------
# Visualize Predictions
# --------------------------
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
axes = axes.flatten()

tick_positions = np.arange(0, 16, 2)

for class_label in range(4):
    # Find all test indices of the current class
    class_indices = np.where(Y_test == class_label)[0]
    
    # Select 2 random indices for this class
    selected_indices = np.random.choice(class_indices, size=2, replace=False)

    for row in range(2):
        idx = selected_indices[row]
        img = X_test[idx]
        img_disp = img.squeeze() if img.ndim == 3 else img

        logits = model(img, params, weights, biases)
        pred_label = int(np.argmax(logits))
        true_label = int(Y_test[idx])

        ax = axes[class_label + row * 4]
        ax.imshow(img_disp, cmap='gray')
        ax.set_title(f"True: {true_label}, Pred: {pred_label}", fontsize=8)

        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(axis='both', labelsize=6)

fig.suptitle("PennyLane QML", fontsize=14)
plt.tight_layout()
plt.savefig("./output/pennylane_qml_test_images_pred_by_class.png")
plt.close()

# --------------------------
# Confusion Matrix
# --------------------------

# Predict all test samples
Y_test_preds = []
for x in X_test:
    logits = model(x, params, weights, biases)
    pred_label = int(np.argmax(logits))
    Y_test_preds.append(pred_label)

cm = confusion_matrix(Y_test, Y_test_preds)

custom_cmap = LinearSegmentedColormap.from_list("pinkblue", ["#0043a3", "white"])

fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
disp_plot = disp.plot(ax=ax_cm, cmap=custom_cmap, values_format='d')

disp_plot.im_.set_clim(vmin=0, vmax=50)

plt.title("PennyLane QML")
plt.savefig("./output/pennylane_qml_confusion_matrix.png")
plt.close()