import numpy as np
import matplotlib.pyplot as plt

# Base class for Perceptron
class Perceptron:
    def __init__(self, learning_rate=1e-5, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_values = []

    def initialize_parameters(self, input_size):
        # Initialize weights for each input feature and a single bias value
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

class RegressorPerceptron(Perceptron):
    def __init__(self, learning_rate=1e-5, epochs=100):
        super().__init__(learning_rate, epochs)
    
    def predict(self, X):
        # Predict output for the given input data
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
class BinaryClassifierPerceptron(Perceptron):
    def __init__(self, learning_rate=1e-5, epochs=100):
        super().__init__(learning_rate, epochs)
    
    def predict_proba(self, X):
        # Predict probabilities for the given input data (useful for metrics like ROC AUC)
        X = np.array(X)
        logits = np.dot(X, self.weights) + self.bias
        return self.activation_function(logits)

    def predict(self, X):
        # Predict probabilities for the given input data
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)  # Convert probabilities to binary labels