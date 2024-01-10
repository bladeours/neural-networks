import logging
import math
import sys
import random
from dataclasses import dataclass

logging.getLogger().setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)


@dataclass
class data:
    alpha: float
    beta: float
    x: float
    y: float


@dataclass
class Perceptron:
    number: int
    weights: list[float]

    def __str__(self):
        return f"{self.number}, {self.weights}"

    def initialize_weights(self):
        self.weights = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)]

    def activate(self, x, y) -> float:
        activation = x * self.weights[0] + y * self.weights[1] + 1  # bias na koncu
        return 1 / (1 + math.exp(-activation))


@dataclass
class RobotArm:
    r1: int
    r2: int
    learning_rate: float

    # input_perceptrons: list[Perceptron]  # Warstwa wejściowa
    hidden_perceptrons: list[list[Perceptron]]  # Warstwy ukryte
    output_perceptrons: list[Perceptron]  # Warstwa wyjściowa

    def __init__(self, r1: int, r2: int, num_hidden_layers: int, learning_rate: float):
        self.r1 = r1
        self.r2 = r2
        self.learning_rate = learning_rate

        # Warstwa wejściowa (alpha i beta)
        # self.input_perceptrons = [Perceptron(i, []) for i in range(2)]
        # for perceptron in self.input_perceptrons:
        #     perceptron.initialize_weights()

        # Warstwy ukryte
        self.hidden_perceptrons = []
        for i in range(num_hidden_layers):
            layer = [Perceptron(j, []) for j in range(2)]
            # for perceptron in layer:
            #     perceptron.initialize_weights()
            layer[0].weights = [0.15, 0.20]
            layer[1].weights = [0.25, 0.30]
            self.hidden_perceptrons.append(layer)

        # Warstwa wyjściowa (x i y)
        self.output_perceptrons = [Perceptron(i, []) for i in range(2)]
        for perceptron in self.output_perceptrons:
            perceptron.initialize_weights()
        self.output_perceptrons[0].weights = [0.4, 0.45]
        self.output_perceptrons[1].weights = [0.50, 0.55]

    def __str__(self):
        return (
            f"{self.r1}, {self.r2}, hidden: {self.hidden_perceptrons}, outputs: {self.output_perceptrons}"
        )

    def calculate_end_point(self, alpha, beta) -> (float, float):
        beta = (beta - (90 - alpha)) * math.pi / 180
        alpha = (alpha - 90) * math.pi / 180
        x = self.r1 * math.cos(alpha)
        y = -self.r1 * math.sin(alpha)
        x = x - self.r2 * math.cos(beta)
        y = y + self.r2 * math.sin(beta)
        return round(x, 10), round(y, 10)

    def forward_pass_and_back(self, x: float, y: float) -> ((float, float), (float, float)):
        # HIDDEN LAYER 1
        h1, h2 = self.predict(self.hidden_perceptrons[0][0], self.hidden_perceptrons[0][1], x, y)

        # OUTPUT LAYER
        o1, o2 = self.predict(self.output_perceptrons[0], self.output_perceptrons[1], h1, h2)
        return (o1, o2), (h1, h2)

    def predict(self, perceptron_alfa: Perceptron, perceptron_beta: Perceptron, x: float, y: float):
        result_0 = perceptron_alfa.activate(x, y)
        result_1 = perceptron_beta.activate(x, y)

        return result_0, result_1

    def calculate_error(self, pred_a: float, pred_b: float, exp_a: float, exp_b: float) -> float:
        error_alpha = 0.5 * ((exp_a - pred_a) ** 2)
        error_beta = 0.5 * ((exp_b - pred_b) ** 2)
        return error_alpha + error_beta

    def update_weights_output_layer(
        self, output_a: float, output_b: float, target_a: float, target_b: float, output_h1: float, output_h2: float
    ):
        # https://theneuralblog.com/forward-pass-backpropagation-example/
        # O1
        ## W5
        result = (output_a - target_a) * (output_a * (1 - output_a)) * output_h1
        w5 = self.output_perceptrons[0].weights[0]
        w5 = w5 - self.learning_rate * result
        self.output_perceptrons[0].weights[0] = w5

        ## W6
        result = (output_a - target_a) * (output_a * (1 - output_a)) * output_h2
        w6 = self.output_perceptrons[0].weights[1]
        w6 = w6 - self.learning_rate * result
        self.output_perceptrons[0].weights[1] = w6

        # O2
        ## W7
        result = (output_b - target_b) * (output_b * (1 - output_b)) * output_h1
        w7 = self.output_perceptrons[1].weights[0]
        w7 = w7 - self.learning_rate * result
        self.output_perceptrons[1].weights[0] = w7

        ## W8
        result = (output_b - target_b) * (output_b * (1 - output_b)) * output_h2
        w8 = self.output_perceptrons[1].weights[1]
        w8 = w8 - self.learning_rate * result
        self.output_perceptrons[1].weights[1] = w8

    def update_weights_hidden_layer(
        self,
        output_a: float,
        output_b: float,
        target_a: float,
        target_b: float,
        output_h1: float,
        output_h2: float,
        input_a: float,
        input_b: float,
    ):
        # H1
        ## W1
        result = (
            ((output_a - target_a) * (output_a * (1 - output_a)) * self.output_perceptrons[0].weights[0])
            * (output_h1 * (1 - output_h1))
            * input_a
        )
        w1 = self.hidden_perceptrons[0][0].weights[0]
        w1 = w1 - self.learning_rate * result
        self.hidden_perceptrons[0][0].weights[0] = w1

        ## W2
        result = (
            ((output_a - target_a) * (output_a * (1 - output_a)) * self.output_perceptrons[0].weights[1])
            * (output_h1 * (1 - output_h1))
            * input_b
        )
        w2 = self.hidden_perceptrons[0][0].weights[1]
        w2 = w2 - self.learning_rate * result
        self.hidden_perceptrons[0][0].weights[1] = w2

        # H2
        ## W3
        result = (
            ((output_b - target_b) * (output_b * (1 - output_b)) * self.output_perceptrons[1].weights[0])
            * (output_h2 * (1 - output_h2))
            * input_a
        )
        w3 = self.hidden_perceptrons[0][1].weights[0]
        w3 = w3 - self.learning_rate * result
        self.hidden_perceptrons[0][1].weights[0] = w3

        ## W4
        result = (
            ((output_b - target_b) * (output_b * (1 - output_b)) * self.output_perceptrons[1].weights[1])
            * (output_h2 * (1 - output_h2))
            * input_b
        )
        w4 = self.hidden_perceptrons[0][1].weights[1]
        w4 = w4 - self.learning_rate * result
        self.hidden_perceptrons[0][1].weights[1] = w4


# test_date: list[data] = []
learning_rate = 0.05
robot = RobotArm(5, 5, 2, learning_rate)
# for i in range(1000):
#     alpha = (random.randint(0, 180)/180.0) * 0.9
#     beta = (random.randint(0, 180)/180.0) * 0.9
#     x, y = robot.calculate_end_point(alpha, beta)
#     test_date.append(data(alpha, beta, x, y))

# Utwórz listy do przechowywania wartości funkcji kosztu w trakcie uczenia
errors = []





(output_a, output_b), (hidden_a, hidden_b) = robot.forward_pass_and_back(0.05, 0.10)
    # Obliczenie błędu dla warstwy wyjściowej
error_output = robot.calculate_error(output_a, output_b, 0.01, 0.99)

# Aktualizacja wag w warstwie wyjściowej
robot.update_weights_output_layer(output_a, output_b, 0.01, 0.99, output_a, output_b)
# Aktualizacja wag w warstwach ukrytych na podstawie błędu wyjściowego
robot.update_weights_hidden_layer(output_a, output_b, 0.01, 0.99, hidden_a, hidden_b, 0.05, 0.10)



for i in range(100_000):
    rand = random.randint(0, len(test_date) - 1)
    alpha = test_date[rand].alpha
    beta = test_date[rand].beta
    x = test_date[rand].x
    y = test_date[rand].y
    
    # Przekazanie danych do sieci i propagacja w przód
    (output_a, output_b), (hidden_a, hidden_b) = robot.forward_pass_and_back(x, y)
    # Obliczenie błędu dla warstwy wyjściowej
    error_output = robot.calculate_error(output_a, output_b, alpha, beta)
    
    # Aktualizacja wag w warstwie wyjściowej
    robot.update_weights_output_layer(output_a, output_b, alpha, beta, output_a, output_b)
    # Aktualizacja wag w warstwach ukrytych na podstawie błędu wyjściowego
    robot.update_weights_hidden_layer(output_a, output_b, alpha, beta, hidden_a, hidden_b, x, y)

    if i % 100 == 0:
        print(f"Iteration {i}, Current Error: {error_output}")


