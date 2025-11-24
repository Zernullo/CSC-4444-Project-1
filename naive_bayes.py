# Author: Daniel Guo
# Reference: ChatGPT, Gemini
from matplotlib import pyplot as plt
import numpy as np
from utils import load_mnist
"""
Naïve Bayes classifier implementation for MNIST dataset.
Naïve Bayes is a probabilistic classifier based on Bayes' theorem with the assumption of feature independence.
In simple terms, it calculates the probability of each class given the input features and assigns the class with the highest probability.
So using the MNIST dataset, we estimate the probability of each pixel being on (1) or off (0) for each digit class (0-9) from the training data.
In this model we trained on 48000 samples and tested on 12000 samples from MNIST dataset to determine the accuracy of Naïve Bayes.
"""
# Train the Naïve Bayes model, estimating pixel probabilities for each digit class
def train_naive_bayes(X_train, y_train):
    X_bin = (X_train > 0.5).astype(np.uint8) # Binarize pixel values: pixels > 0.5 set to 1, else 0
    probs = {} # to store pixel probabilities for each digit class
    for digit in range(10): # loop over each digit class (0-9), to estimate pixel probabilities
        X_digit = X_bin[y_train == digit] # select all training samples belonging to the current digit class
        probs[digit] = (X_digit.sum(axis=0) + 1) / (len(X_digit) + 2) # Laplace smoothing to estimate P(pixel=1 | class=digit), meaning for each pixel, count how many times it is 1 in samples of this digit class, add 1 for smoothing, divide by total samples of this class + 2
    return probs

# Predict labels for test data using the trained Naïve Bayes model, using log probabilities for numerical stability
def predict_naive_bayes(probs, X_test): 
    X_bin = (X_test > 0.5).astype(np.uint8) # Binarize test pixel values, same as training data, pixels > 0.5 set to 1, else 0
    y_pred = [] # to store predicted labels for test samples
    for x in X_bin: # loop over each test sample to predict its label
        scores = [np.sum(np.log(p) * x + np.log(1 - p) * (1 - x)) for p in probs.values()] # compute log-probability scores for each digit class using the learned pixel probabilities, for numerical stability we use log probabilities: log(P(class)) + sum over pixels [log(P(pixel=1|class)) if pixel=1 else log(P(pixel=0|class))]. In simple terms, for each pixel in the test sample, if the pixel is 1 we add log(probability of pixel being 1 for that class), else we add log(probability of pixel being 0 for that class)
        y_pred.append(np.argmax(scores)) # assign the class with the highest log-probability score as the predicted label
    return np.array(y_pred)

if __name__ == "__main__":
    X, y = load_mnist('MNIST', flatten=True, normalize=True) # load MNIST data, flatten images to vectors and normalize pixel values, utils return x as images and y as labels
    np.random.seed(42) # The purpose of seed is to ensure reproducibility of the random operations that following, in simple terms it makes sure that every time you run the code you get the same random numbers. Seed of 42 is just a commonly used arbitrary number.
    indices = np.arange(len(X)) # create an array of indices from 0 to number of samples - 1. len(X) gives total number of samples in the dataset so this creates an array [0, 1, 2, ..., 59999] for MNIST which has 60000 samples
    np.random.shuffle(indices) # shuffle the indices randomly to ensure random splitting of data into training and test sets
    split = int(0.8 * len(X)) # determine the split index for 80% training and 20% testing
    X_train, y_train = X[indices[:split]], y[indices[:split]] # first 80% of shuffled data for training
    X_test, y_test = X[indices[split:]], y[indices[split:]] # remaining 20% for testing

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    nb_probs = train_naive_bayes(X_train, y_train) # train the Naïve Bayes model to learn pixel probabilities for each digit class
    y_pred = predict_naive_bayes(nb_probs, X_test) # predict labels for test data using the trained Naïve Bayes model
    acc = (y_pred == y_test).mean() # compute accuracy by comparing predicted labels with true labels
    print(f"Naïve Bayes Accuracy: {acc:.4f}")

    plt.figure(figsize=(10,4))
    for digit in range(10):
        plt.subplot(2,5,digit+1)
        plt.imshow(nb_probs[digit].reshape(28,28), cmap='gray')
        plt.title(f"Digit {digit}")
        plt.axis('off')
    plt.suptitle("Naïve Bayes Learned Pixel Probabilities")
    plt.show()

# To run Naïve Bayes, in the terminal use: python naive_bayes.py