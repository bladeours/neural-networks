from dataclasses import dataclass

import cv2 as cv
import random
import os
import logging, sys


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
    logging.getLogger().setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


def get_files(full_path: str) -> list[tuple[int, str]]:
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
            result.append(int(str(p).replace("0", "1").replace("255", "-1")))
    return Image(i, result)


def get_image_vector_from_list(img: list[list[int]]):
    result = []
    for row in img:
        result += row
    return Image(1, result)


def get_training_images() -> list:
    path = "trainImages"
    
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


def train_perceptron(train_images: list[Image], perceptron: Perceptron, learning_rate):
    error_counter = 0
    logging.debug(f"training {perceptron.number}...")
    max_lifetime = 0
    current_lifetime = 0

    for _ in range(15_000):
        rand = random.randint(0, len(train_images) - 1)
        train_image = train_images[rand]
        output = predict(train_image, perceptron)
        expected_output = 1 if train_image.number == perceptron.number else -1
        err = expected_output - output
        if err == 0:
            current_lifetime += 1
            if current_lifetime > max_lifetime:
                max_lifetime = current_lifetime
        else:
            current_lifetime = 0
            error_counter += 1
            perceptron.weights = [w + learning_rate * err * x for w, x in zip(perceptron.weights, train_image.data)]
    logging.debug(f"errors: {error_counter}, max lifetime: {max_lifetime}")


def train_all_perceptrons(train_images: list[Image], perceptrons: list[Perceptron], learning_rate):
    for perceptron in perceptrons:
        train_perceptron(train_images, perceptron, learning_rate)
