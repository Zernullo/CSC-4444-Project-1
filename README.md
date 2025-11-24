The project creates 5 machine learning models for digit recognition. We compare all 5 models to 
determine the complexity of the training. The KNN model compares a validation image to 
training images and assigns the label of the most similar ones based on pixel distance, the 
accuracy of my result was between 96-97% for k = 1,3,5. Naive Bayes model assumes each 
pixel value is independent and estimates probabilities for each pixel to each digit, the accuracy 
of my result was 83%. Linear Classification was trying to find a single straight hyperplane for 
each 0-9 data, the accuracy of my result was 84%. Multilayer Perceptron is a connected 
network with hidden layers and a nonlinear activation function which is ReLU, the accuracy of 
my result was 90%. CNN uses convolutional layers to process images, the accuracy of my 
result was 95
