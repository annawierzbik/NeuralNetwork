# Simple Neural Network in Python

This repository contains a simple feedforward neural network implemented from scratch in **Python** using **NumPy**. The network supports multiple layers, different activation functions, and a basic training loop for classification tasks.

---

## Features

* Fully connected layers (Dense)
* Activation functions:

  * Sigmoid
  * Tanh
  * ReLU
  * Softmax
* Cross-entropy loss for multi-class classification
* Adjustable learning rate
* Configurable number of layers and epochs
* Accuracy tracking during training
* Basic interface for visualizing and recognizing digits

---

## Installation

No external deep learning libraries are required. Only **NumPy**, **TensorFlow** (for reproducible random seeds), and **Matplotlib** (for visualization) are used:

```bash
pip install numpy tensorflow matplotlib
```

---

## Usage

### 1. Define the network

```python
from network import Network, FullyConnected, ReLu, Sigmoid, Tanh, Softmax, Loss, cross_entropy_loss, cross_entropy_loss_derivative

layers = [
    FullyConnected(input_size=784, output_size=128, init_type=0),
    ReLu(),
    FullyConnected(input_size=128, output_size=64, init_type=0),
    Tanh(),
    FullyConnected(input_size=64, output_size=10, init_type=0),
    Softmax()
]

network = Network(layers=layers, learning_rate=0.01)
loss = Loss(loss_function=cross_entropy_loss, loss_function_derivative=cross_entropy_loss_derivative)
network.compile(loss=loss)
```

### 2. Train the network

```python
network.fit(x_train, y_train, epochs=50, learning_rate=0.01, verbose=1)
```

* `x_train` — input data (NumPy array)
* `y_train` — one-hot encoded labels
* `epochs` — number of training iterations
* `learning_rate` — learning rate for gradient descent
* `verbose` — how often to print progress

### 3. Evaluate the network

```python
accuracy = network.accuracy(x_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
```

### 4. Visualize training progress

```python
import matplotlib.pyplot as plt

plt.plot(network.epoch_list, network.losses, label='Loss')
plt.plot(network.epoch_list, network.accuracies, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()
```

---

## Customization

* Change the **number of layers**, **neurons per layer**, and **activation functions** to experiment with different architectures.
* Adjust **learning rate** and **number of epochs** for faster or more stable convergence.
* Supports simple digit recognition tasks, but for better accuracy, consider adding more layers or using convolutional layers.

---

## Notes

* This implementation is educational and **not optimized for performance**.
* It works best with small datasets or simple experiments.
* Softmax layer's backward pass is simplified to work with cross-entropy loss.

---

## License

MIT License
