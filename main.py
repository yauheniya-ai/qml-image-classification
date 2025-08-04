import subprocess

print("\n\nStarting Classical ML...")
subprocess.run(["python", "image_classification_classical.py"], check=True)

print("\n\nStarting PennyLane QML...")
subprocess.run(["python", "image_classification_pennylane.py"], check=True)

print("\n\nStarting Qiskit QML...")
subprocess.run(["python", "image_classification_qiskit.py"], check=True)
