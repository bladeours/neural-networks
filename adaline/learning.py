from cgi import test
from dataclasses import dataclass
from lib2to3.fixes.fix_import import traverse_imports
import re
from cv2 import correctMatches
from keras.datasets import mnist
import math
import random
import logging, sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import double


@dataclass
class Perceptron:
    number: int
    weights: list[float]

    def __str__(self):
        return f"{self.number}, {self.weights}"


@dataclass
class Image:
    number: int
    data: list


def setup_logger():
    logging.getLogger().setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


def get_image(i: int, image: list[list[int]]) -> Image:
    result = []
    result.append(1)
    for row in image:
        for p in row:
            p //= 127
            result.append(int(str(p).replace("0", "-1")))
    return Image(i, result)



def get_training_images() -> list[Image]:
    #dodac bias
    images = []
    logging.warning("loading training data...")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    for data, number in zip(train_X, train_y):
        images.append(get_image(number, data.tolist()))
    return images


def get_test_images() -> list[Image]:
    images = []
    logging.warning("loading test data...")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    for data, number in zip(test_X, test_y):
        images.append(get_image(number, data.tolist()))
    return images
    


def create_perceptrons(train_images: list[Image]) -> list[Perceptron]:
    perceptrons = []
    used = []
    for train_image in train_images:
        if train_image.number in used:
            continue
        else:
            used.append(train_image.number)
        weights = []
        for _ in train_image.data:
            weights.append(random.uniform(-0.1, 0.1))
        perceptron = Perceptron(train_image.number, weights)
        perceptrons.append(perceptron)
    return perceptrons


def predict(image: Image, perceptron: Perceptron) -> float:
    sum = 0
    for w,x in zip(perceptron.weights, image.data):
        sum += w * x
    return sum



def train_perceptron(train_images: list[Image], perceptron: Perceptron, learning_rate) -> Perceptron:
    avg_err = 0
    errors = []
    for i in range(100_000):
        rand = random.randint(0, len(train_images) - 1)
        train_image = train_images[rand]
        output = predict(train_image, perceptron)
        expected_output = 1 if train_image.number == perceptron.number else -1
        err = expected_output - output
        avg_err += abs(err)
        perceptron.weights = [weight + learning_rate * err * data for weight, data in zip(perceptron.weights, train_image.data)]
        if i%10000 == 0:
            logging.warning(f"training {perceptron.number}, i: {i}, error: {avg_err/10000}")
            errors.append(avg_err/10000)
            avg_err = 0
    xpoints = np.array(errors[1:])
    plt.plot(xpoints)
    plt.show()
    return perceptron

def train_all_perceptrons(train_images: list[Image], perceptrons: list[Perceptron], learning_rate) -> list[Perceptron]:
    perceptrons_new = []
    for perceptron in perceptrons:
        perceptron = train_perceptron(train_images, perceptron, learning_rate)
        perceptrons_new.append(perceptron)
    return perceptrons_new

def check_if_learned(perceptrons: list[Perceptron], test_images: list[Image]) -> int: 
    correct_counter = 0
    for test_image in test_images:
        max = 0
        perceptron_number = 0
        for perceptron in perceptrons:
            result = predict(test_image, perceptron) 
            if(result > max):
                max = result
                perceptron_number = perceptron.number
        if(test_image.number == perceptron_number):
            correct_counter += 1
            
    return correct_counter

