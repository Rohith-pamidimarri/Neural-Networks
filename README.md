# Handwritten Digit Recognition using Artificial Neural Networks

## Introduction

This project presents a handmade implementation of an Artificial Neural Network (ANN) for recognizing handwritten digits from the MNIST dataset. The model is built entirely from scratch, without relying on external libraries, showcasing a deep understanding of neural network principles.

## Overview

In this project, a subset of the MNIST dataset containing 3,750 training images and 1,250 test images of handwritten digits is utilized. The primary objective is to train a neural network to accurately classify these digits into their respective classes (0 through 9). Key components of the project include:

1. **Data Preparation**: Loading the dataset, preprocessing the images, and transforming them into a suitable format for training. This involves normalization and reshaping of the images.

2. **Model Building**: Designing the neural network architecture, including the input layer, hidden layers with activation functions, and the output layer with softmax activation for multi-class classification. Notably, the model is constructed entirely from scratch, without utilizing pre-existing deep learning libraries.

3. **Training**: Training the model using gradient descent optimization and implementing both feedforward and backpropagation algorithms. Forward propagation computes the output, while backpropagation calculates gradients for updating the weights and biases.

4. **Evaluation**: Evaluating the trained model on a separate test set to measure its accuracy in classifying handwritten digits.

## Model Building

### Data Preparation

The dataset is loaded and preprocessed, ensuring that the images are in a suitable format for training. Pixel values are normalized, and images are reshaped as necessary.

### Model Architecture

The neural network architecture is handcrafted, with careful consideration given to the number of layers, neurons, and activation functions. The implementation includes feedforward and backpropagation algorithms, showcasing a deep understanding of neural network fundamentals.

### Training

The model is trained using gradient descent optimization, with both feedforward and backpropagation algorithms implemented from scratch. This approach allows for a comprehensive understanding of how neural networks learn from data.

### Evaluation

The trained model is evaluated on a separate test set to assess its performance in classifying handwritten digits. The achieved accuracy rate of approximately 90% underscores the effectiveness of the handmade neural network implementation.

## Conclusion

This project underscores the power of implementing neural networks from scratch, without relying on external libraries. By constructing the model using feedforward and backpropagation techniques, a deeper understanding of neural network principles is attained. Moving forward, further experimentation and refinement can be pursued to enhance the model's performance.

