import logging
import math
import sys
import random
from dataclasses import dataclass
import numpy as np

def normalize_data(number, min_value, max_value) -> float:
    normalized = ((number - min_value) * 2) / (max_value - min_value) - 1
    return normalized

def reverse_normalize_data(normalized, min_value, max_value) -> float:
    reversed_val = ((normalized + 1) * (max_value - min_value) / 2) + min_value
    return reversed_val

@dataclass
class RobotArm:
    r1: int
    r2: int

    def __str__(self):
        return (
            f"{self.r1}, {self.r2}"
        )

    def calculate_end_point(self, alpha, beta) -> (float, float):
        beta = (beta - (90 - alpha)) * math.pi / 180
        alpha = (alpha - 90) * math.pi / 180
        x = self.r1 * math.cos(alpha)
        y = -self.r1 * math.sin(alpha)
        x = x - self.r2 * math.cos(beta)
        y = y + self.r2 * math.sin(beta)
        return round(x, 10), round(y, 10)

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
		# initialize the list of weights matrices, then store the
		# network architecture and learning rate
        self.weights = []
        self.layers = layers
        self.alpha = alpha
        for i in range(len(layers) - 2):
            # adding an extra node for the bias
            w = np.random.uniform(-0.1, 0.1, size=(layers[i] + 1, layers[i + 1] + 1))
            self.weights.append(w / np.sqrt(layers[i]))
            w = np.random.uniform(-0.1, 0.1, size=(layers[-2] + 1, layers[-1]))
            self.weights.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
  
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def train(self, X, y, epochs=1000, displayUpdate=100):
        # insert a column of 1's as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train
            # our network on it
            # rand = random.randint(0, len(X) - 1)
            
            for (x, target) in zip(X, y):
                # self.train_partial(X[rand], y[rand])
                self.train_partial(x, target)
                # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
                    
    def train_partial(self, x, target):
		# construct our list of output activations for each layer
		# as our data point flows through the network; the first
		# activation is a special case -- it's just the input
		# feature vector itself
        A = [np.atleast_2d(x)]
        		# FEEDFORWARD:
		# loop over the layers in the network
        for layer in np.arange(0, len(self.weights)):
			# feedforward the activation at the current layer by
			# taking the dot product between the activation and
			# the weight matrix -- this is called the "net input"
			# to the current layer
            net = A[layer].dot(self.weights[layer])
			# computing the "net output" is simply applying our
			# nonlinear activation function to the net input
            out = self.sigmoid(net)
			# once we have the net output, add it to our list of
			# activations
            A.append(out)
            		# BACKPROPAGATION
		# the first phase of backpropagation is to compute the
		# difference between our *prediction* (the final output
		# activation in the activations list) and the true target
		# value
        output = A[-1]
        difference = output - target
		# from here, we need to apply the chain rule and build our
		# list of deltas 'D'; the first entry in the deltas is
		# simply the error of the output layer times the derivative
		# of our activation function for the output value
        D = [difference * self.sigmoid_deriv(output)]
        		# once you understand the chain rule it becomes super easy
		# to implement with a 'for' loop -- simply loop over the
		# layers in reverse order (ignoring the last two since we
		# already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
			# the delta for the current layer is equal to the delta
			# of the *previous layer* dotted with the weight matrix
			# of the current layer, followed by multiplying the delta
			# by the derivative of the nonlinear activation function
			# for the activations of the current layer
            delta = D[-1].dot(self.weights[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
          		# since we looped over our layers in reverse order we need to
		# reverse the deltas
        D = D[::-1]
		# WEIGHT UPDATE PHASE
		# loop over the layers
        for layer in np.arange(0, len(self.weights)):
			# update our weights by taking the dot product of the layer
			# activations with their respective deltas, then multiplying
			# this value by some small learning rate and adding to our
			# weight matrix -- this is where the actual "learning" takes
			# place
            self.weights[layer] += -self.alpha * A[layer].T.dot(D[layer])
            
            
    def predict(self, X, addBias=True):
		# initialize the output prediction as the input features -- this
		# value will be (forward) propagated through the network to
		# obtain the final prediction
        p = np.atleast_2d(X)
		# check to see if the bias column should be added
        if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]
		# loop over our layers in the network
        for layer in np.arange(0, len(self.weights)):
			# computing the output prediction is as simple as taking
			# the dot product between the current activation value 'p'
			# and the weight matrix associated with the current layer,
			# then passing this value through a nonlinear activation
			# function
            p = self.sigmoid(np.dot(p, self.weights[layer]))
		# return the predicted value
        return p
    
    def calculate_loss(self, X, targets):
		# make predictions for the input data points then compute
		# the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
		# return the loss
        return loss
            
robot = RobotArm(5,5)
maxValue = robot.r1 + robot.r2 
input = []
exp_output = []
for i in range(100):
    alpha = random.randint(0, 180)
    beta = random.randint(0, 180)
    x, y = robot.calculate_end_point(alpha, beta)
    x = normalize_data(x, -maxValue, maxValue)
    y = normalize_data(y, -maxValue, maxValue)
    alpha = normalize_data(alpha, 0, 180)
    beta = normalize_data(beta, 0, 180)
    input.append([x,y])
    exp_output.append([alpha, beta])
    

input = np.array(input)
exp_output = np.array(exp_output)
train_data = [input, exp_output]

# input = []
# exp_output = []
# for i in range(100):
#     alpha = (random.randint(0, 180)/180.0) * 0.9
#     beta = (random.randint(0, 180)/180.0) * 0.9
#     x, y = robot.calculate_end_point(alpha, beta)
#     input.append([x,y])
#     exp_output.append([alpha, beta])
# input = np.array(input)
# exp_output = np.array(exp_output)
# test_data = [input, exp_output]

nn = NeuralNetwork([2, 16, 2], alpha=0.2)

nn.train(train_data[0], train_data[1], epochs=100_000)   

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])
# nn = NeuralNetwork([2, 2, 1], alpha=0.5)
# nn.fit(X, y, epochs=20000)
# exp_output = exp_output
# for (x, target) in zip(input, exp_output):
# 	# make a prediction on the data point and display the result
# 	# to our console
# 	pred = nn.predict(x)[0][0]
# 	print("[INFO] data={}, target={}, pred={:.4f}".format(
# 		x, target, pred))