from ast import Str
from dataclasses import dataclass
from itertools import permutations
import re

import cv2 as cv
import random
import os
import logging, sys


@dataclass
class Perceptron:
    number: int
    weights: list

    def __str__(self):
        return f"{self.number}, {self.weights}"


@dataclass
class Image:
    number: int
    data: list

def setup_logger():
    logging.getLogger().setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    

def get_files(full_path: str) -> list[str]:
    files = []
    for path in os.listdir(full_path):
        joined_path = os.path.join(full_path, path)
        if os.path.isfile(joined_path):
            files.append((int(path[0]), joined_path))
    return files


def get_image_vector(i: int, file: str) -> Image:
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    img_list = img.tolist()
    result = []
    for row in img_list:
        for p in row:
            result.append(int(str(p).replace("0", "-1")))
    return Image(i, result)


def get_training_images() -> list:
    path = "C:\\Users\\a826510\\Desktop\\neural-network\\neural-networks\\zad_1\\trainImages"
    numbers = []
    for i, file in get_files(path):
        numbers.append(get_image_vector(int(i), file))
    return numbers

def get_perceptrons(train_images: list[Image]) -> list[Perceptron]:
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


def predict(image: Image, perceptron: Perceptron) -> int:
    result = sum(w * x for w, x in zip(perceptron.weights, image.data))
    output = 1 if result > 0 else -1
    return output



def train_perceptron(train_images: list[Image], perceptron: Perceptron, learning_rate) -> int:
    error_counter = 0
    logging.debug(f"training {perceptron.number}...")
    for _ in range(15_000):
        rand = random.randint(0, len(train_images)-1)
        train_image = train_images[rand]
        output = predict(train_image, perceptron)
        expected_output = 1 if train_image.number == perceptron.number else -1
        err = expected_output - output
        if err == 0:
            continue
        error_counter += 1
        perceptron.weights = [w + learning_rate * err * x for w, x in zip(perceptron.weights, train_image.data)]
    return error_counter

setup_logger()
train_images = get_training_images()
perceptrons = get_perceptrons(train_images)

learning_rate = 0.1

for perceptron in perceptrons:
    logging.debug(f"{perceptron.number} errors:{train_perceptron(train_images, perceptron, learning_rate)}")
    
logging.debug("after training:")
for perceptron in perceptrons:
    logging.debug(f"{perceptron.number} errors:{train_perceptron(train_images, perceptron, learning_rate)}")
    
    
for perceptron in perceptrons:
    output = predict(get_image_vector(1, 'C:\\Users\\a826510\\Desktop\\neural-network\\neural-networks\\zad_1\\toCheck\\toCheck.jpg'), perceptron) 
    print(f"perceptron: {perceptron.number}\t output: {output}")

