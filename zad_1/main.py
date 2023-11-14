from dataclasses import dataclass

import cv2 as cv
import random


@dataclass
class Perceptron:
    number: int
    weights: list

    def __str__(self):
        return f"{self.number}, {self.weights}"


def getBinaryImages(files: list) -> list:
    numbers = []
    for file in files:
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img //= 255
        img_list = img.tolist()
        result = []
        for row in img_list:
            for p in row:
                result.append(int(str(p).replace('0', '-1')))
        numbers.append(result)
    return numbers


def getPerceptrons(numbers: list) -> list:
    perceptrons = []
    for i, number in enumerate(numbers):
        weights = []
        for pixel in number[0]:
            weights.append(random.uniform(-0.1, 0.1))
        perceptron = Perceptron(i, weights)
        perceptrons.append(perceptron)
    return perceptrons


def trainPerceptrons(number: int, data: list, perceptron: Perceptron, learning_rate: float):
    print(f"training {number}...")
    for i in range(15_000):
        result = sum(w * x for w, x in zip(perceptron.weights, data))
        output = 1 if result > 0 else -1
        expected_output = 1 if number == perceptron.number else -1
        err = expected_output - output
        if err == 0:
            return
        perceptron.weights = [w + learning_rate * err * x for w, x in zip(perceptron.weights, data)]



numbers = [getBinaryImages(["0_1.jpg", "0_2.jpg", "0_3.jpg", "0_4.jpg", "0_5.jpg"]),
           getBinaryImages(["1_1.jpg", "1_2.jpg", "1_3.jpg", "1_4.jpg", "1_5.jpg"]),
           getBinaryImages(["2_1.jpg", "2_2.jpg", "2_3.jpg", "2_4.jpg", "2_5.jpg"]),
           getBinaryImages(["3_1.jpg", "3_2.jpg", "3_3.jpg", "3_4.jpg", "3_5.jpg"])]

perceptrons = getPerceptrons(numbers)
learning_rate = 0.1
trainPerceptrons(0, numbers[0][0], perceptrons[0], learning_rate)
