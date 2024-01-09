import logging
import math
import sys
import random
from dataclasses import dataclass

@dataclass
class Perceptron:
    number: int
    weights: list[float]

    def __str__(self):
        return f"{self.number}, {self.weights}"
    
    def initialize_weights(self):
        self.weights = [1, random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]

    def activate(self, inputs):
        #funkcja sigmoidalna
        activation = sum(w * input_value for w, input_value in zip(self.weights, inputs))
        return 1 / (1 + math.exp(-activation)) 
    
@dataclass
class RobotArm:
    r1: int
    r2: int
    
    input_perceptrons: list[Perceptron]  # Warstwa wejściowa
    hidden_perceptrons: list[Perceptron]  # Warstwy ukryte
    output_perceptrons: list[Perceptron]  # Warstwa wyjściowa

    def __init__(self, r1: int, r2: int, num_hidden_layers: int, ):
        self.r1 = r1
        self.r2 = r2

        # Warstwa wejściowa (alpha i beta)
        self.input_perceptrons = [Perceptron(i, []) for i in range(2)]
        for perceptron in self.input_perceptrons:
            perceptron.initialize_weights()  # Inicjalizacja wag

        # Warstwy ukryte
        self.hidden_perceptrons = []
        for i in range(num_hidden_layers):
            layer = [Perceptron(j, []) for j in range(2)]
            for perceptron in layer:
                perceptron.initialize_weights()  # Inicjalizacja wag
            self.hidden_perceptrons.append(layer)

        # Warstwa wyjściowa (x i y)
        self.output_perceptrons = [Perceptron(i, []) for i in range(2)]
        for perceptron in self.output_perceptrons:
            perceptron.initialize_weights()  # Inicjalizacja wag

    def __str__(self):
        return f"{self.r1}, {self.r2}, inputs: {self.input_perceptrons}, " \
               f"hidden: {self.hidden_perceptrons}, outputs: {self.output_perceptrons}"

    def calculate_end_point(self, alpha, beta) -> (float, float):
        beta = (beta - (90 - alpha)) * math.pi/180;
        alpha = (alpha - 90) * math.pi/180;
        x = self.r1 * math.cos(alpha)
        y = -self.r1 * math.sin(alpha)
        x = x - self.r2 * math.cos(beta)
        y = y + self.r2 * math.sin(beta)
        return round(x, 10), round(y, 10)

    def predict_end_point(self, alpha: float, beta: float) -> (float, float):
        alpha_inputs = [1, alpha, beta]
        beta_inputs = [1, alpha, beta]

        # Forward pass przez warstwę ukrytą
        hidden_outputs = [neuron.activate(alpha_inputs + beta_inputs) for neuron in self.hidden_layer]

        # Forward pass do warstwy wyjściowej
        x = self.alpha_perceptron.activate([1] + hidden_outputs)
        y = self.beta_perceptron.activate([1] + hidden_outputs)

        predicted_end_point = (x, y)
        return predicted_end_point


def setup_logger():
    logging.getLogger().setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

robot = RobotArm(5, 5, 1)
alpha = math.pi  # 45 stopni
beta = math.pi / 2  # 60 stopni

print(robot)