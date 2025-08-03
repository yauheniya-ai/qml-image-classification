import time
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector 
import numpy as np
import pennylane as qml  # needed just for loading the dataset
import matplotlib.pyplot as plt

plt.style.use('dark_background') 

# Load the "Plus Minus" dataset from PennyLane
ds = qml.data.load("plus-minus")[0]
X_train, Y_train = ds.img_train, ds.labels_train
X_test, Y_test = ds.img_test, ds.labels_test

# One-hot encoding
def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

Y_train_oh = one_hot(Y_train, 4)
Y_test_oh = one_hot(Y_test, 4)

# Expectation helper for Z on wires 0 and 1
def expectation_z(state, wire):
    z_op = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)
    obs = 1
    for i in range(8):
        obs = np.kron(obs, z_op if i == wire else identity)
    return np.real(np.dot(state.conj().T, obs @ state))

# Create parameterized circuit
def create_circuit(image, params):
    qc = QuantumCircuit(8)

    amps = np.array(image).flatten()
    norm = np.linalg.norm(amps)
    amps_norm = amps / norm if norm != 0 else amps
    
    # Ensure proper normalization with higher precision
    amps_norm = amps_norm / np.linalg.norm(amps_norm)
    
    # Use Qiskit's built-in normalize parameter instead of manual normalization
    qc.initialize(amps_norm, range(8), normalize=True)

    for i in range(4):
        for j in range(8):
            qc.ry(params[i, j], j)
        for j in range(7):
            qc.cx(j, j+1)
    return qc

# Forward pass: run circuit and get expectation values
def get_expvals(image, params):
    qc = create_circuit(image, params)
    sv = Statevector.from_instruction(qc)
    state = sv.data
    return np.array([expectation_z(state, 0), expectation_z(state, 1)])

# Model output
def model(image, params, weights, biases):
    expvals = get_expvals(image, params)
    logits = weights @ expvals + biases
    return logits

# Cross-entropy loss with softmax
def cross_entropy(y_true, logits):
    exps = np.exp(logits - np.max(logits))
    softmax = exps / np.sum(exps)
    return -np.sum(y_true * np.log(softmax + 1e-10))

# Loss over batch
def batch_loss(params, weights, biases):
    total = 0
    for x, y in zip(X_train, Y_train_oh):
        logits = model(x, params, weights, biases)
        total += cross_entropy(y, logits)
    return total / len(X_train)

# Accuracy
def accuracy(params, weights, biases, X, Y):
    correct = 0
    for x, y in zip(X, Y):
        logits = model(x, params, weights, biases)
        pred = np.argmax(logits)
        if pred == y:
            correct += 1
    return correct / len(Y)

# Initialize parameters
np.random.seed(0)
params = np.random.normal(0, np.pi, size=(4, 8))
weights = np.random.normal(0, 0.1, size=(4, 2))
biases = np.zeros(4)

# Gradient-free optimization (e.g., simple SGD)
steps = 300  # Adjust the number of steps
lr = 0.3     # Set learning rate

print(f"Starting training for {steps} steps...")
start_time = time.time()

losses = []
train_accuracies = []
step_times = []

for step in range(steps):
    step_start = time.time()
    
    grad_params = np.zeros_like(params)
    grad_weights = np.zeros_like(weights)
    grad_biases = np.zeros_like(biases)

    eps = 5e-4  # Smaller epsilon for more accurate gradients

    # Estimate gradients (numerical)
    base_loss = batch_loss(params, weights, biases)
    
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            params_perturbed = np.copy(params)
            params_perturbed[i, j] += eps
            grad_params[i, j] = (batch_loss(params_perturbed, weights, biases) - base_loss) / eps

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights_perturbed = np.copy(weights)
            weights_perturbed[i, j] += eps
            grad_weights[i, j] = (batch_loss(params, weights_perturbed, biases) - base_loss) / eps

    for i in range(len(biases)):
        biases_perturbed = np.copy(biases)
        biases_perturbed[i] += eps
        grad_biases[i] = (batch_loss(params, weights, biases_perturbed) - base_loss) / eps

    # Gradient descent step
    params -= lr * grad_params
    weights -= lr * grad_weights
    biases -= lr * grad_biases

    step_time = time.time() - step_start
    
    # Show progress every step since training is slow
    acc = accuracy(params, weights, biases, X_train, Y_train)
    elapsed_total = time.time() - start_time

    losses.append(base_loss)
    train_accuracies.append(acc)
    step_times.append(step_time)

    print(f"Step: {step+1:2d}/{steps}, Loss: {base_loss:.4f}, Train acc: {acc:.4f}, Step time: {step_time/60:.2f}min, Total elapsed: {elapsed_total/60:.1f}min")

total_time = time.time() - start_time
print(f"\nTraining completed in {total_time/60:.1f} minutes.")
print(f"Average time per step: {total_time/steps/60:.1f} minutes.")

test_acc = accuracy(params, weights, biases, X_test, Y_test)
print(f"Test accuracy: {test_acc:.4f}")

######## Visualizations ########

# --------------------------
# Plot Accuracy & Loss
# --------------------------

# Plot training accuracy and loss curves side-by-side
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Plot accuracy curve (single line)
axes[0].plot(train_accuracies, label='Train Accuracy', c="#52bcff")
axes[0].set_title('Accuracy over Steps')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Plot loss curve (single line)
axes[1].plot(losses, label='Train Loss', c="#ea2081")
axes[1].set_title('Loss over Steps')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.2)

fig.suptitle("Qiskit QML", fontsize=14)

plt.tight_layout()
plt.savefig("./output/qiskit_qml_training_accuracy_loss.png")  # Save the plot locally
plt.close()

# --------------------------
# Visualize Predictions
# --------------------------

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
axes = axes.flatten()

tick_positions = np.arange(0, 16, 2)

for class_label in range(4):
    # Find indices in the test set with the current class label
    class_indices = np.where(Y_test == class_label)[0]
    
    # Randomly pick 2 samples from this class
    selected_indices = np.random.choice(class_indices, size=2, replace=False)

    for row in range(2):
        idx = selected_indices[row]
        img = X_test[idx]
        img_disp = img.squeeze() if img.ndim == 3 else img

        logits = model(img, params, weights, biases)
        pred_label = np.argmax(logits)
        true_label = Y_test[idx]

        ax = axes[class_label + row * 4]
        ax.imshow(img_disp, cmap='gray')
        ax.set_title(f"True: {true_label}, Pred: {pred_label}", fontsize=8)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions.astype(int))
        ax.set_yticklabels(tick_positions.astype(int))
        ax.tick_params(axis='both', labelsize=6)

fig.suptitle("Qiskit QML", fontsize=14)

plt.tight_layout()
plt.savefig("./output/qiskit_qml_test_images_pred.png")
plt.close()

