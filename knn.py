# Author: Daniel Guo
# Reference: ChatGPT, Gemini
from matplotlib import pyplot as plt
import numpy as np
from utils import load_mnist
import time

"""
K-Nearest Neighbors (KNN) classifier implementation for MNIST dataset.
KNN is a simple, instance-based learning algorithm that classifies a sample based on the majority label of its k nearest neighbors in the training set.
In simple terms, it looks at the closest training examples to decide the label of a new example.
So using the MNIST dataset, we calculate the Euclidean distance between the test samples and all training samples to find the nearest neighbors.
In this model we trained on 48000 samples and tested on 12000 samples from MNIST dataset to determine the accuracy of KNN with different k values.
"""
# X_train and y_train are the training data and labels, X_test is the test data, k is the number of neighbors, batch_size is for processing test data in chunks to save memory and speed
def knn_predict(X_train, y_train, X_test, k=3, batch_size=100): 
    num_test = X_test.shape[0] # number of test samples. shape[0] gives the number of rows meaning number of samples in which case their are 12000 test samples
    y_pred = np.empty(num_test, dtype=int) # initialize an empty array to store predicted labels for test samples

    X_train_sq = np.sum(X_train**2, axis=1)  # shape (48000), precompute squared norms of training samples meaning sum of squares of each training sample. There are 48000 training samples
    
    for i in range(0, num_test, batch_size): # process test samples in batches so that we don't run out of memory and speed up computation
        X_batch = X_test[i:i+batch_size]
        X_batch_sq = np.sum(X_batch**2, axis=1).reshape(-1, 1)  # shape (batch, 1), this means sum of squares of each test sample in the batch
        dists = X_batch_sq + X_train_sq - 2 * (X_batch @ X_train.T)  # shape (batch, 48000), squared Euclidean distances using the formula ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        nearest = np.argsort(dists, axis=1)[:, :k] # shape (batch, k), get indices of k nearest neighbors for each test sample in the batch
        y_pred[i:i+batch_size] = [np.bincount(y_train[n]).argmax() for n in nearest] # majority vote among the k nearest neighbors to assign label to each test sample in the batch
        print(f"Processed {i+len(X_batch)} / {num_test} test samples") 
    return y_pred

if __name__ == "__main__":
    X, y = load_mnist('MNIST', flatten=True, normalize=True) # load MNIST data, flatten images to vectors and normalize pixel values, utils return x as images and y as labels
    np.random.seed(42) # The purpose of seed is to ensure reproducibility of the random operations that following, in simple terms it makes sure that every time you run the code you get the same random numbers. Seed of 42 is just a commonly used arbitrary number.
    indices = np.arange(len(X)) # create an array of indices from 0 to number of samples - 1. len(X) gives total number of samples in the dataset so this creates an array [0, 1, 2, ..., 59999] for MNIST which has 60000 samples
    np.random.shuffle(indices) # shuffle the indices randomly to ensure random splitting of data into training and test sets
    split = int(0.8 * len(X)) # determine the split index for 80% training and 20% testing
    X_train, y_train = X[indices[:split]], y[indices[:split]] # first 80% of shuffled data for training
    X_test, y_test = X[indices[split:]], y[indices[split:]] # remaining 20% for testing

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    ks = [1, 3, 5] # different k values to try for KNN
    accuracies = [] # to store accuracies for different k values
    for k in ks: # loop over different k values to determine which k gives best accuracy
        start = time.time()
        y_pred = knn_predict(X_train, y_train, X_test, k=k, batch_size=100)
        acc = (y_pred == y_test).mean()
        accuracies.append(acc)
        print(f"KNN Accuracy (k={k}): {acc:.4f}, Time: {time.time()-start:.1f}s")
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(ks, accuracies, marker='o', color='blue')
    plt.title("KNN Accuracy vs k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.xticks(ks)
    plt.grid(True)
    plt.show()

# To run KNN, in the terminal use: python knn.py
