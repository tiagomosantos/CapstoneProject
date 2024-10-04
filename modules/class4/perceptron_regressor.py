import numpy as np
import matplotlib.pyplot as plt
from perceptron import RegressorPerceptron

# Subclass for Stochastic Gradient Descent
class RegressorPerceptronSGD(RegressorPerceptron):
    def __init__(self, learning_rate=1e-5, epochs=100, shuffle=True):
        super().__init__(learning_rate, epochs)
        self.shuffle = shuffle

    def train(self, X, y):
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Initialize parameters based on the number of features in the input
        self.initialize_parameters(X.shape[1])

        self.epoch_info_loss = {}

        for epoch in range(self.epochs):
            if self.shuffle:
                # Shuffle the data
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]

            self.epoch_info_loss[epoch] = {}

            epoch_losses = []

            for i in range(X.shape[0]):  # Loop through each sample
                x_i = X[i]
                y_i = y[i]

                # Compute the linear combination of weights and inputs + bias
                y_pred = np.dot(x_i, self.weights) + self.bias

                # Calculate the error
                error = y_i - y_pred

                # Update weights and bias using the Delta Rule
                self.weights += self.learning_rate * error * x_i
                self.bias += self.learning_rate * error

                # Calculate Mean Squared Error (MSE) for the current sample
                sample_loss = error ** 2

                # Store the loss for the current sample
                epoch_losses.append(sample_loss)

            # Calculate the average and standard deviation of the loss for the current epoch
            self.epoch_info_loss[epoch]['mean'] = np.mean(epoch_losses)
            self.epoch_info_loss[epoch]['std'] = np.std(epoch_losses)

            # Calculate Mean Squared Error (MSE) for the entire training set
            y_pred_all = np.dot(X, self.weights) + self.bias
            loss = np.mean((y - y_pred_all) ** 2)
            self.loss_values.append(loss)


    def plot_loss(self):
        # Create a plot for the batch loss with epochs as the x-axis
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values, label='Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_epoch_info_loss(self):
        # Create a plot for the loss of each sample in each epoch
        plt.figure(figsize=(10, 5))
        for epoch in self.epoch_info_loss:
            mean = self.epoch_info_loss[epoch]['mean']
            std = self.epoch_info_loss[epoch]['std']
            plt.errorbar(epoch, mean, yerr=std, fmt='o', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss per Sample in each Epoch')
        plt.grid()
        plt.show()

# Subclass for Batch Gradient Descent
class RegressorPerceptronBGD(RegressorPerceptron):
    def __init__(self, learning_rate=1e-5, epochs=100):
        super().__init__(learning_rate, epochs)

    def train(self, X, y):
        """Train the model using Batch Gradient Descent."""
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Initialize parameters based on the number of features in the input
        self.initialize_parameters(X.shape[1])

        for epoch in range(self.epochs):
            # Predict for the entire dataset
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate error
            error = y - y_pred

            # Multiple each error position by the corresponding input feature
            inside_sum = np.dot(X.T, error)
            
            # Calculate the weigth update for the entire batch
            weight_update = self.learning_rate * inside_sum.sum(axis=1) * 2 / len(X)
  
            # Update weights and bias using the batch gradient
            self.weights += weight_update
            self.bias += self.learning_rate * np.mean(error)

            # Calculate Mean Squared Error (MSE) for the entire training set
            y_pred_all = np.dot(X, self.weights) + self.bias
            loss = np.mean((y - y_pred_all) ** 2)
            self.loss_values.append(loss)

            # Print loss every epoch for monitoring
            if (epoch + 1) % 1 == 0:  # Adjust printing frequency if needed
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss:.4f}')

    def plot_loss(self):
        # Plot the loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values, color='green', label=' Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.grid()
        plt.show()

# Subclass for Mini-Batch Gradient Descent with an optional Shuffling parameter
class RegressorPerceptronMBGB(RegressorPerceptron):
    def __init__(self, learning_rate=1e-5, epochs=100, batch_size=16, shuffle=True):
        super().__init__(learning_rate, epochs)
        self.batch_size = batch_size
        self.shuffle = shuffle  # New parameter to control shuffling

    def train(self, X, y):
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Initialize parameters based on the number of features in the input
        self.initialize_parameters(X.shape[1])

        self.epoch_info_loss = {}  # Store loss values for each epoch

        for epoch in range(self.epochs):
            # Shuffle the data at the beginning of each epoch if shuffle is set to True
            if self.shuffle:
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]

            self.epoch_info_loss[epoch] = {}
            epoch_losses = []

            # Loop through mini-batches
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Compute the linear combination of weights and inputs + bias
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Calculate the error
                error = y_batch - y_pred

                # Multiple each error position by the corresponding input feature
                inside_sum = np.dot(X_batch.T, error)
                
                # Calculate the weight update for the entire batch
                weight_update = self.learning_rate * inside_sum.sum(axis=1) * 2 / len(X_batch)
    
                # Update weights and bias using the batch gradient
                self.weights += weight_update
                self.bias += self.learning_rate * np.mean(error)

                # Calculate Mean Squared Error (MSE) for the batch
                y_pred_batch = np.dot(X_batch, self.weights) + self.bias
                loss = np.mean((y_batch - y_pred_batch) ** 2)
                
                epoch_losses.append(loss)   

            # Calculate the average and standard deviation of the loss for the current epoch
            self.epoch_info_loss[epoch]['mean'] = np.mean(epoch_losses)
            self.epoch_info_loss[epoch]['std'] = np.std(epoch_losses)

            # Calculate Mean Squared Error (MSE) for the entire training set
            y_pred_all = np.dot(X, self.weights) + self.bias
            loss = np.mean((y - y_pred_all) ** 2)
            self.loss_values.append(loss)

            # Print loss every epoch for monitoring
            if (epoch + 1) % 1 == 0:  # Adjust printing frequency if needed
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss:.4f}')


    def plot_loss(self):
        # Plot the loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values, label=' Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_epoch_info_loss(self):
        # Create a plot for the loss of each sample in each epoch
        plt.figure(figsize=(10, 5))
        for epoch in self.epoch_info_loss:
            mean = self.epoch_info_loss[epoch]['mean']
            std = self.epoch_info_loss[epoch]['std']
            plt.errorbar(epoch, mean, yerr=std, fmt='o', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title('Loss per Batch in each Epoch')
        plt.grid()
        plt.show()