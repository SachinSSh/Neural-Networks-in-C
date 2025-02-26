# Neural Network for MNIST Classification in C

A from-scratch implementation of a neural network in C to classify handwritten digits from the MNIST dataset. This project includes data loading, network training, and evaluation components.

![MNIST Sample](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Overview
This repository contains a simple feedforward neural network implemented in C. It trains on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of 28x28 grayscale handwritten digit images (0-9). The implementation demonstrates:

- MNIST data loading and preprocessing
- Neural network initialization and training
- Forward/backward propagation
- Activation functions (ReLU and Softmax)
- Mini-batch gradient descent optimization

Designed for educational purposes to understand neural network fundamentals and low-level implementation details.

## Features
- **MNIST Data Loader**: Reads raw MNIST files into memory
- **Network Architecture**: Configurable input/hidden/output layers
- **Activation Functions**: ReLU (hidden layer) and Softmax (output)
- **Cross-Entropy Loss**: With numerical stability measures
- **Training Loop**: Mini-batch gradient descent support
- **Evaluation**: Accuracy calculation on test set

## Prerequisites
- C compiler (GCC/Clang)
- GNU Make
- Python 3 (for data conversion script)
- MNIST dataset files (included in setup)

## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/nn-in-c-mnist.git
   cd nn-in-c-mnist
