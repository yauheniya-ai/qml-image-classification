import subprocess

print("Starting Classical ML...")
subprocess.run(["python", "im_class_classical.py"], check=True)

print("Starting PennyLane QML...")
subprocess.run(["python", "im_class_pennylane.py"], check=True)

print("Starting Qiskit QML...")
subprocess.run(["python", "im_class_qiskit.py"], check=True)
