# Neural Network from Scratch in Python

## Overview
This project implements a simple neural network from scratch using NumPy. The network supports multiple layers, activation functions, and backpropagation for training. It includes:

- Fully connected layers
- Activation functions: Tanh, ReLU, Sigmoid, Softmax
- Cross-entropy loss
- A training loop with backpropagation

## Features
- **Custom Layer Implementation**: Supports fully connected layers.
- **Activation Functions**: Includes Tanh, ReLU, Sigmoid, and Softmax.
- **Backpropagation**: Implements gradient descent-based learning.
- **Loss Function**: Uses cross-entropy loss for classification tasks.
- **Training**: Supports batch training with accuracy tracking.

## Classes & Functions
### Layer (Abstract Base Class)
Defines the structure for all neural network layers.

- `forward(x)`: Propagates input forward.
- `backward(output_error_derivative)`: Propagates error backward.

### FullyConnected Layer
Implements a fully connected (dense) layer with weights and biases.

- `forward(x)`: Computes output using `x * weights + biases`.
- `backward(output_error)`: Updates weights and biases using backpropagation.

### Activation Functions
#### Tanh
- `forward(x)`: Applies the Tanh function.
- `backward(output_error)`: Computes derivative of Tanh.

#### ReLU
- `forward(x)`: Applies ReLU (Rectified Linear Unit).
- `backward(output_error)`: Computes derivative of ReLU.

#### Sigmoid
- `forward(x)`: Applies the Sigmoid function.
- `backward(output_error)`: Computes derivative of Sigmoid.

#### Softmax
- `forward(x)`: Applies the Softmax function.
- `backward(output_error)`: Passes the error backward.

### Loss Functions
#### Cross-Entropy Loss
- `cross_entropy_loss(y_pred, y_true)`: Computes categorical cross-entropy loss.
- `cross_entropy_loss_derivative(y_pred, y_true)`: Computes derivative for backpropagation.

### Network Class
Defines the neural network with forward and backward propagation.

- `compile(loss)`: Assigns a loss function.
- `fit(x_train, y_train, epochs, learning_rate)`: Trains the network.
- `accuracy(x_test, y_test)`: Computes classification accuracy.
  
---
Made by Anna Wierzbik, Michał Iwanow-Kołakowski

