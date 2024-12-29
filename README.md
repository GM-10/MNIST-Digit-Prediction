# MNIST Digit Prediction Model
This repository contains a Jupyter Notebook (MNIST_Digits_Classification.ipynb) that implements a machine learning model for classifying handwritten digits using the classic MNIST dataset.

## Project Goal:

Train a model to accurately identify handwritten digits (0-9) from the MNIST dataset.
## MNIST Dataset:

The MNIST dataset is a widely used benchmark for image classification tasks. It consists of 70,000 images of handwritten digits, each labeled with its corresponding value (0-9).

## Model Architecture:

This project utilizes a common approach for image classification: Convolutional Neural Networks (CNNs). CNNs are adept at extracting features from images, making them well-suited for tasks like digit recognition.

## Jupyter Notebook:

The MNIST_Digits_Classification.ipynb notebook walks you through the following steps:

### Data Loading and Preprocessing:
Loads the MNIST dataset using library: keras from TensorFlow.
Preprocesses the images by converting them to grayscale, reshaping and normalizing pixel values.
### Converting Labels to One-Hot encoding
It convert categorical data into a numerical format suitable for machine learning algorithms.
### Model Building:
First make the normal ANN with fully connected layers then define a CNN architecture with convolutional layers and pooling layers. 
Configures the model with hyperparameters like learning rate and optimizer.
### Model Training:
Trains the model on a portion of the MNIST dataset using backpropagation and gradient descent optimization.
Monitors training progress using metrics like accuracy and loss.

### Visualizing The Training Process
In this section, I have visualized the model accuracy and model loss.

### Model Evaluation:
Evaluates the trained model's performance on a separate test set from the MNIST dataset.

### Visualizing the Errors in form of Confusion Matrix
This assesses the model's generalization ability

### Predicting the Errors for 10 different digits 
In this section, I have identified the most unpredicted numbers and arranged the numbers in asceding order of errors associated with it.


## Getting Started:

Install Dependencies: Ensure you have the necessary libraries like  Keras, NumPy, and Matplotlib installed. (You can install them using pip install keras numpy matplotlib in your terminal.)</br>
Run the Notebook: Open the MNIST_Digits_Classification.ipynb notebook in Google Colab or a local Jupyter Notebook environment.</br>
Execute Cells: Run the cells in the notebook sequentially to load data, build the model, train it, and evaluate its performance.</br>



## Results:
Test Loss: 0.026102764531970024</br>
Test_accuracy:0.9922999739646912 </br>
