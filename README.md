# Fashion-MNIST CNN Classifier with Optuna Hyperparameter Tuning

This project demonstrates how to build a convolutional neural network (CNN) for classifying images from the Fashion-MNIST dataset using PyTorch, with hyperparameter optimization via Optuna. The notebook includes data preprocessing, augmentation, training, evaluation, and automated tuning of model hyperparameters.

## 📂 Project Structure
.
├── Fashion-MNIST-CNN.ipynb   # Jupyter notebook containing the full code
├── fashion-mnist_train.csv   # Fashion-MNIST CSV dataset
├── README.md                 # Project overview

## 🛠 Features

Custom PyTorch Dataset for handling Fashion-MNIST images.

Data Augmentation: Random rotation, horizontal flip, and affine transformations.

Flexible CNN Architecture: Configurable number of convolutional and fully connected layers, filter sizes, kernel sizes, and dropout rates.

Hyperparameter Optimization: Uses Optuna to search for the best combination of:

Learning rate

Optimizer type (SGD, Adam, RMSprop)

Batch size

Number of layers

Dropout rate

Weight decay

Training & Evaluation:

Tracks training loss, training accuracy, and test accuracy.

Checks for overfitting by comparing train and test accuracy per epoch.

Device Agnostic: Automatically detects and uses GPU (CUDA or Apple MPS) if available, else CPU.

## 📦 Dependencies

Python 3.8+

PyTorch
torchvision
pandas
scikit-learn
matplotlib
optuna

Install dependencies via pip:
pip install torch torchvision pandas scikit-learn matplotlib optuna

## 🚀 Usage

Clone the repository:
  git clone https://github.com/your-username/fashion-mnist-cnn.git
  cd fashion-mnist-cnn
Open the Jupyter Notebook:
Run all cells to:

Load and preprocess the dataset

Perform data augmentation

Optimize hyperparameters using Optuna

Train the final CNN model

Evaluate the model on the test set

Observe the train and test accuracy per epoch to check for overfitting.

## 📈 Results

The notebook prints training loss, training accuracy, and test accuracy at each epoch.

Optuna automatically finds the hyperparameters that maximize validation accuracy.

