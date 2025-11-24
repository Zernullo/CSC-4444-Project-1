# Author: Daniel Guo
# Reference: ChatGPT, Gemini
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_mnist

"""
MLP (Multi-Layer Perceptron) implementation for MNIST dataset.
An MLP is a type of feedforward artificial neural network that consists of multiple layers of nodes.
In simple terms, it processes input data through several layers, applying weights and activation functions to learn complex patterns.
So using the MNIST dataset, we learn weights for each pixel in the image through multiple layers to classify the digits (0-9).
In this model we trained on 48000 samples and tested on 12000 samples from MNIST dataset to determine the accuracy of the MLP.
"""
# Define the MLP architecture. The purpose of this class is to create a neural network model with two hidden layers for digit classification.
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10): # input_size is 784 for 28x28 images flattened, hidden1 and hidden2 are sizes of the two hidden layers, num_classes is 10 for digits 0-9
        super().__init__() # initialize the parent class nn.Module
        self.model = nn.Sequential( # create a sequential model, a sequential model is a linear stack of layers
            nn.Linear(input_size, hidden1), # what is nn.Linear? nn.Linear is a fully connected layer that applies a linear transformation to the incoming data: y = xA^T + b where A is weight matrix and b is bias vector. creates a fully connected layer from input_size to hidden1
            nn.ReLU(), # nn.ReLU applies the ReLU activation function. ReLU stands for Rectified Linear Unit and is defined as f(x) = max(0, x). It introduces non-linearity to the model, allowing it to learn complex patterns.
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    # Perform forward pass through the network. This method defines how the input data flows through the network layers to produce an output.
    def forward(self, x): 
        return self.model(x)

# Train the MLP model using gradient descent
def train_mlp(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64, lr=0.01):
    X_train = torch.tensor(X_train, dtype=torch.float32) # torch tensor is a multi-dimensional array used in PyTorch for building and training neural networks. Here we convert the training data to a tensor of type float32.
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True) # create a data loader for batching and shuffling the training data
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class. It is used for multi-class classification problems. This is used to compute the loss between the predicted outputs and true labels.
    optimizer = optim.SGD(model.parameters(), lr=lr) # Stochastic Gradient Descent (SGD) optimizer to update model parameters based on computed gradients. Optimizer adjusts the weights of the model to minimize the loss function.

    test_acc_list = []  # store test accuracy per epoch

    for epoch in range(epochs): # loop over the number of epochs to train the model
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward() # backward is used to compute the gradient of the loss with respect to the model parameters. It performs backpropagation, calculating how much each parameter contributed to the loss.
            optimizer.step() # step is used to update the model parameters based on the computed gradients. It applies the optimization algorithm (in this case, SGD) to adjust the weights and biases of the model.
        
        # Evaluate test accuracy at the end of the epoch
        with torch.no_grad():
            pred = torch.argmax(model(X_test), dim=1) # get predicted labels by taking the class with highest score
            acc = (pred == y_test).float().mean().item() # compute accuracy by comparing predicted labels with true labels
            test_acc_list.append(acc) # append the accuracy to the list
        print(f"Epoch {epoch+1}, Test Accuracy: {acc:.4f}")
    return test_acc_list

if __name__ == "__main__":
    X, y = load_mnist('MNIST', flatten=True, normalize=True) # load MNIST data, flatten images to vectors and normalize pixel values, utils return x as images and y as labels
    np.random.seed(42) # The purpose of seed is to ensure reproducibility of the random operations that following, in simple terms it makes sure that every time you run the code you get the same random numbers. Seed of 42 is just a commonly used arbitrary number.
    indices = np.arange(len(X)) # create an array of indices from 0 to number of samples - 1. len(X) gives total number of samples in the dataset so this creates an array [0, 1, 2, ..., 59999] for MNIST which has 60000 samples
    np.random.shuffle(indices) # shuffle the indices randomly to ensure random splitting of data into training and test sets
    split = int(0.8 * len(X)) # determine the split index for 80% training and 20% testing
    X_train, y_train = X[indices[:split]], y[indices[:split]] # first 80% of shuffled data for training
    X_test, y_test = X[indices[split:]], y[indices[split:]] # remaining 20% for testing

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    model = MLP() # initialize the MLP model
    test_acc_list = train_mlp(model, X_train, y_train, X_test, y_test, epochs=5) # train the model for 5 epochs

    # Plot test accuracy vs epoch
    epochs = list(range(1, len(test_acc_list)+1))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, test_acc_list, marker='o', color='green')
    plt.title("MLP Test Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

# To run MLP, in the terminal use: python mlp.py