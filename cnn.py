# Author: Daniel Guo
# Reference: ChatGPT, Gemini
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_mnist
import numpy as np

"""
CNN (Convolutional Neural Network) implementation for MNIST dataset.
A CNN is a type of deep learning model particularly effective for image data, as it can automatically and adaptively learn spatial hierarchies of features through convolutional layers.
In simple terms, it uses filters to scan over the image and capture important patterns like edges, shapes, and textures that help in recognizing digits.
So using the MNIST dataset, we learn filters that help identify different digit classes (0-9) from the pixel data.
In this model we trained on 48000 samples and tested on 12000 samples from MNIST dataset to determine the accuracy of the CNN.
"""
# Define the CNN architecture. The purpose of this class is to create a convolutional neural network model for digit classification.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), # Conv2D is a convolutional layer that applies a set of filters to the input image. Here, we have 1 input channel (grayscale), 32 output channels (filters), a kernel size of 3x3, and padding of 1 to maintain the spatial dimensions.
            nn.ReLU(),
            nn.MaxPool2d(2), # MaxPool2D is a pooling layer that reduces the spatial dimensions of the feature maps by taking the maximum value in each 2x2 window, effectively downsampling the image.
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# Train the CNN model using gradient descent
def train_cnn(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64, lr=0.01):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=lr)

    test_acc_list = []  # store test accuracy per epoch

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        # Evaluate test accuracy at the end of each epoch
        with torch.no_grad():
            pred = torch.argmax(model(X_test), dim=1)
            acc = (pred == y_test).float().mean().item()
            test_acc_list.append(acc)
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
    X_train = X_train.reshape(-1,1,28,28) # reshape to (batch_size, channels, height, width). reshape is used to change the shape of the numpy array without changing its data. Here we reshape the training data to have 1 channel and 28x28 height and width for CNN input.
    X_test = X_test.reshape(-1,1,28,28)

    model = CNN() # initialize the CNN model
    test_acc_list = train_cnn(model, X_train, y_train, X_test, y_test, epochs=5) # train the model for 5 epochs
    
    # Plot Test Accuracy vs Epoch
    epochs = list(range(1, len(test_acc_list)+1))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, test_acc_list, marker='o', color='purple')
    plt.title("CNN Test Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()

# To run CNN, in the terminal use: python cnn.py