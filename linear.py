# Author: Daniel Guo
# Reference: ChatGPT, Gemini
from matplotlib import pyplot as plt
import numpy as np
from utils import load_mnist

"""
Linear Classifier implementation for MNIST dataset.
A Linear Classifier is a simple model that makes predictions based on a linear combination of input features.
In simple terms, it assigns weights to each pixel in the image and combines them to decide which digit the image represents.
So using the MNIST dataset, we learn weights for each pixel that help distinguish between different digit classes (0-9).
In this model we trained on 48000 samples and tested on 12000 samples from MNIST dataset to determine the accuracy of the Linear Classifier.
"""
# Define the Linear Classifier
class LinearClassifier:
    # Initialize the model parameters, self stands for the instance of the class, input_dim is the number of input features (pixels), num_classes is the number of output classes (digits 0-9), lr is learning rate, reg is regularization strength
    def __init__(self, input_dim, num_classes, lr=0.1, reg=1e-4):
        self.W = 0.01 * np.random.randn(num_classes, input_dim) # weights initialized to small random values for symmetry breaking
        self.b = np.zeros((num_classes, 1)) # biases initialized to zero to start without any initial bias
        self.lr = lr # learning rate controls how much to change the model parameters during training
        self.reg = reg # regularization strength helps prevent overfitting by penalizing large weights

    # Softmax function to convert logits to probabilities. What softmax does is it takes a vector of raw scores (logits) and transforms them into probabilities that sum to 1. This is useful for multi-class classification problems like MNIST where we have 10 classes (digits 0-9).
    def softmax(self, z): # The parameter z is a 2D numpy array of shape (num_classes, num_samples) representing the raw scores (logits) for each class and each sample. This means that each column in z corresponds to a different sample, and each row corresponds to a different class.
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True)) # subtracting max for numerical stability. This prevents very large exponentials which can lead to overflow errors.
        return exp_z / np.sum(exp_z, axis=0, keepdims=True) # normalize to get probabilities. This divides each exponentiated score by the sum of all exponentiated scores for that sample, ensuring that the probabilities for each sample sum to 1.

    # Train the Linear Classifier using gradient descent
    def train(self, X, y, epochs=10): # X is the training data of shape (num_samples, input_dim), y is the true labels of shape (num_samples,), epochs is the number of training iterations
        X_mean = X.mean(axis=0) # compute mean for normalization to ensure each feature has zero mean. axis 0 means we are computing the mean across all samples for each feature (pixel)
        X_std = X.std(axis=0) + 1e-8 # compute std for normalization to ensure each feature has unit variance, adding a small constant (1e-8) to avoid division by zero. std means standard deviation
        X = (X - X_mean) / X_std # normalize the data to have zero mean and unit variance. This helps with convergence during training

        X = X.T.astype(np.float32) # transpose X to shape (input_dim, num_samples) for matrix operations and convert to float32 for efficiency. transpose X means swapping its rows and columns. X.T changes the shape from (num_samples, input_dim) to (input_dim, num_samples). X.T.astype(np.float32) ensures that the data type of the array is float32, which is more memory efficient and faster for computations compared to the default float64.
        Y = np.eye(10, dtype=np.float32)[y].T # one-hot encode labels to shape (num_classes, num_samples). np.eye(10) creates a 10x10 identity matrix where each row corresponds to a one-hot encoded vector for each class (digit 0-9). By indexing this matrix with y, we get the one-hot encoded representation of the labels. The .T at the end transposes the resulting array to have shape (num_classes, num_samples). one-hot encoding is a way to represent categorical variables as binary vectors. For example, if we have 3 classes (0, 1, 2), the one-hot encoding would represent class 0 as [1, 0, 0], class 1 as [0, 1, 0], and class 2 as [0, 0, 1].
        losses = [] # to store loss values for each epoch

        for epoch in range(epochs): # loop over the number of epochs to train the model
            Z = self.W @ X + self.b # compute logits, shape (num_classes, num_samples). The @ operator performs matrix multiplication. Here, self.W has shape (num_classes, input_dim) and X has shape (input_dim, num_samples), so the result Z has shape (num_classes, num_samples). We then add the bias self.b which is broadcasted across all samples.
            A = self.softmax(Z) # apply softmax to get probabilities, shape (num_classes, num_samples)

            loss = -np.mean(np.sum(Y * np.log(A + 1e-8), axis=0)) + self.reg * np.sum(self.W**2) # compute cross-entropy loss with L2 regularization. The first term computes the average cross-entropy loss between the true labels Y and the predicted probabilities A. We add a small constant (1e-8) inside the log to prevent log(0). The second term is the L2 regularization term which penalizes large weights to help prevent overfitting. 
            losses.append(loss) # We append the loss to the losses list for tracking.

            dZ = A - Y # compute gradient of loss w.r.t. (with respect to) logits, shape (num_classes, num_samples). This is to find out how much we need to change the logits Z to reduce the loss. The gradient dZ tells us the direction and magnitude of change needed for each logit.
            dW = (dZ @ X.T) / X.shape[1] + 2 * self.reg * self.W # compute gradient w.r.t. weights, shape (num_classes, input_dim). Here, we multiply dZ with the transpose of X to get the gradient with respect to the weights. We then average it over all samples by dividing by the number of samples (X.shape[1]). The second term adds the gradient of the L2 regularization.
            db = np.mean(dZ, axis=1, keepdims=True) # compute gradient w.r.t. biases, shape (num_classes, 1). We average dZ over all samples to get the gradient for the biases.

            self.W -= self.lr * dW # update weights using gradient descent. We subtract the product of the learning rate and the gradient from the current weights to move in the direction that reduces the loss.
            self.b -= self.lr * db # update biases using gradient descent. Similar to weights, we update the biases by moving in the direction that reduces the loss.

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        return losses, X_mean, X_std

    # Predict labels for new data
    def predict(self, X, X_mean=None, X_std=None): # X is the test data of shape (num_samples, input_dim), X_mean and X_std are for normalization
        if X_mean is not None and X_std is not None: # normalize test data using training mean and std if provided
            X = (X - X_mean) / X_std 
        X = X.T.astype(np.float32) # transpose X to shape (input_dim, num_samples) and convert to float32
        Z = self.W @ X + self.b # compute logits, shape (num_classes, num_samples)
        A = self.softmax(Z) # apply softmax to get probabilities, shape (num_classes, num_samples)
        return np.argmax(A, axis=0)


if __name__ == "__main__":
    X, y = load_mnist('MNIST', flatten=True, normalize=True) # load MNIST data, flatten images to vectors and normalize pixel values, utils return x as images and y as labels
    np.random.seed(42) # The purpose of seed is to ensure reproducibility of the random operations that following, in simple terms it makes sure that every time you run the code you get the same random numbers. Seed of 42 is just a commonly used arbitrary number.
    indices = np.arange(len(X)) # create an array of indices from 0 to number of samples - 1. len(X) gives total number of samples in the dataset so this creates an array [0, 1, 2, ..., 59999] for MNIST which has 60000 samples
    np.random.shuffle(indices) # shuffle the indices randomly to ensure random splitting of data into training and test sets
    split = int(0.8 * len(X)) # determine the split index for 80% training and 20% testing
    X_train, y_train = X[indices[:split]], y[indices[:split]] # first 80% of shuffled data for training
    X_test, y_test = X[indices[split:]], y[indices[split:]] # remaining 20% for testing
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    clf = LinearClassifier(784, 10, lr=0.1, reg=1e-4) # initialize the Linear Classifier with input dimension 784 (28x28 images flattened), 10 classes (digits 0-9), learning rate 0.1 and regularization strength 1e-4
    losses, X_mean, X_std = clf.train(X_train, y_train, epochs=10) # train the model for 10 epochs

    y_pred = clf.predict(X_test, X_mean, X_std) # predict labels for test data using the trained model
    acc = (y_pred == y_test).mean() # compute accuracy by comparing predicted labels with true labels
    print(f"Linear Classifier Accuracy: {acc:.4f}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(range(1, 11), losses, marker='o', color='red')
    plt.title("Linear Classifier Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

# To run Linear Classifier, in the terminal use: python linear.py