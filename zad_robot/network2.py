from dataclasses import dataclass
import random
import math 
import numpy as np


def normalize_data(number, min_value, max_value) -> float:
    normalized = ((number - min_value) * 0.8) / (max_value - min_value) + 0.1
    return normalized

def reverse_normalize_data(normalized, min_value, max_value) -> float:
    reversed_val = ((normalized - 0.1) * (max_value - min_value) / 0.8) + min_value
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

def get_random_weight() -> float:
    return random.uniform(-0.1, 0.1)


def initialize_network(n_inputs, n_hidden, n_outputs):
    # 2 - 1 - 2
    # 2 input 1 hidden 2 output
    #[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
    #[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
    
    network = list()
    hidden_layer = []
    for _ in range(n_hidden):
        weights = {'weights': [get_random_weight() for _ in range(n_inputs + 1)]}
        # weights = {'weights': np.random.normal(0, 1, n_inputs + 1)}
        hidden_layer.append(weights)
    network.append(hidden_layer)
    
    output_layer = []
    for _ in range(n_outputs):
        weights = {'weights':[get_random_weight() for _ in range(n_hidden + 1)]}
        # weights = {'weights': np.random.normal(0, 1,n_hidden + 1)}
        output_layer.append(weights)
    network.append(output_layer)
    return network

# Calculate perceptron activation for an input
def activate(weights, inputs):
    bias = weights[-1]
    activation = 0
    for i in range(len(weights)-1):
       activation += weights[i] * inputs[i]
    return activation + bias

# Transfer perceptron activation
def transfer(activation):
	return 1.0 / (1.0 + math.exp(-activation))

def forward_propagate(network, inputs):
	for layer in network:
		new_inputs = []
		for perceptron in layer:
			activation = activate(perceptron['weights'], inputs)
			perceptron['output'] = transfer(activation)
			new_inputs.append(perceptron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return output * (1.0 - output)

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1: # jezeli layer jest outputem
			for j in range(len(layer)):
				error = 0.0
				for perceptron in network[i + 1]:
					error += (perceptron['weights'][j] * perceptron['delta'])
				errors.append(error)
		else: # jezeli layer jest hidden
			for j in range(len(layer)):
				perceptron = layer[j]
				errors.append(perceptron['output'] - expected[j])
		for j in range(len(layer)):
			perceptron = layer[j]
			perceptron['delta'] = errors[j] * transfer_derivative(perceptron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [perceptron['output'] for perceptron in network[i - 1]]
		for perceptron in network[i]:
			for j in range(len(inputs)):
				perceptron['weights'][j] -= l_rate * perceptron['delta'] * inputs[j]
			perceptron['weights'][-1] -= l_rate * perceptron['delta']

# Train a network for a fixed number of epochs
def train_network(network, dataset, l_rate, n_epoch):
    for epoch in range(n_epoch):
        sum_error = 0
        # rand = random.randint(0, len(dataset) - 1)
        
        for data in dataset:
            outputs = forward_propagate(network, data)
            expected = []
            expected.append(data[-2])
            expected.append(data[-1])
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, data, l_rate)
        if epoch == 0 or epoch%100==0:
            print('epoch=%d, \tlrate=%.3f, \terror=%.3f' % (epoch, l_rate, sum_error))


robot = RobotArm(5,5)
maxValue = robot.r1 + robot.r2 
network = initialize_network(2, 2, 2)

train_dateset = []
for i in range(30):
    alpha = random.randint(0, 180)
    beta = random.randint(0, 180)
    x, y = robot.calculate_end_point(alpha, beta)
    x = normalize_data(x, -maxValue, maxValue)
    y = normalize_data(y, -maxValue, maxValue)
    alpha = normalize_data(alpha, 0, 180)
    beta = normalize_data(beta, 0, 180)
    train_dateset.append([x,y,alpha,beta])

avg_delta_before = 0
avg_alpha_delta_before = 0
avg_beta_delta_before = 0
for data in train_dateset:
    pred_a, pred_b = forward_propagate(network, data)
    pred_a = reverse_normalize_data(pred_a, 0, 180)
    pred_b = reverse_normalize_data(pred_b, 0, 180)
    alpha = reverse_normalize_data(data[2], 0, 180)
    beta = reverse_normalize_data(data[3], 0, 180)
    delta_alpha = abs(pred_a - alpha)
    delta_beta = abs(pred_b - beta)
    delta = delta_alpha + delta_beta
    avg_delta_before += delta
    avg_alpha_delta_before += delta_alpha
    avg_beta_delta_before += delta_beta
    
avg_delta_before = avg_delta_before / len(train_dateset)
avg_beta_delta_before = avg_beta_delta_before / len(train_dateset)
avg_alpha_delta_before = avg_alpha_delta_before / len(train_dateset)

dataset = []
for i in range(100):
    alpha = random.randint(0, 180)
    beta = random.randint(0, 180)
    x, y = robot.calculate_end_point(alpha, beta)
    x = normalize_data(x, -maxValue, maxValue)
    y = normalize_data(y, -maxValue, maxValue)
    alpha = normalize_data(alpha, 0, 180)
    beta = normalize_data(beta, 0, 180)
    dataset.append([x,y,alpha,beta])


train_network(network, dataset, 0.1, 5000)

avg_delta_after = 0
avg_alpha_delta_after = 0
avg_beta_delta_after = 0
for data in train_dateset:
    pred_a, pred_b = forward_propagate(network, data)
    pred_a = reverse_normalize_data(pred_a, 0, 180)
    pred_b = reverse_normalize_data(pred_b, 0, 180)
    alpha = reverse_normalize_data(data[2], 0, 180)
    beta = reverse_normalize_data(data[3], 0, 180)
    alpha_delta = abs(pred_a - alpha)
    beta_delta = abs(pred_b - beta)
    delta = alpha_delta + beta_delta
    avg_delta_after += delta
    avg_alpha_delta_after += alpha_delta
    avg_beta_delta_after += beta_delta
    
avg_delta_after = avg_delta_after / len(train_dateset)
avg_beta_delta_after = avg_beta_delta_after / len(train_dateset)
avg_alpha_delta_after = avg_alpha_delta_after / len(train_dateset)


print(f"before alpha: {avg_alpha_delta_before}, beta: {avg_beta_delta_before} sum: {avg_delta_before}")
print(f"after alpha: {avg_alpha_delta_after}, beta: {avg_beta_delta_after} sum: {avg_delta_after}")
print(f"delta: {avg_delta_before - avg_delta_after}")