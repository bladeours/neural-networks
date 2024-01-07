import matplotlib.pyplot as plt
import numpy as np
from learning import *

    
def perceptron_sort(p: Perceptron):
    return p.number

setup_logger()


train_images = get_training_images()
test_images = get_test_images()

before_list = []
after_list = []


perceptrons = create_perceptrons(train_images)
perceptrons.sort(key=perceptron_sort)
learning_rate = 0.00001

before = check_if_learned(perceptrons, test_images)

perceptrons = train_all_perceptrons(train_images, perceptrons, learning_rate)
logging.warning(f"before: {before}")
after = check_if_learned(perceptrons, test_images)
logging.warning(f"after: {after}")
before_list.append(before / 100)
after_list.append(after / 100)
