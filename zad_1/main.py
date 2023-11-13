import cv2 as cv
import random

class Perceptron:
  def __init__(self, number, weights):
    self.number = number
    self.weights = weights

  def __str__(self):
    return f"{self.number}, {self.weights}"

def getBinaryImages(files):
  numbers = []
  for file in files:
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    img //= 255 
    img_list = img.tolist() 
    result = []
    for row in img_list:
      for p in row:
        result.append(int(str(p).replace('0','-1')))
    numbers.append(result)
  return numbers

    # Read Image


numbers = []
numbers.append(getBinaryImages(["0_1.jpg","0_2.jpg", "0_3.jpg", "0_4.jpg", "0_5.jpg"]))
numbers.append(getBinaryImages(["1_1.jpg","1_2.jpg", "1_3.jpg", "1_4.jpg", "1_5.jpg"]))

perceptrons = []

for i in range(len(numbers)):
  print(i)
  weights = []
  for pixel in numbers[i]:
    print(pixel)
    weights.append(random.random())
  perceptron = Perceptron(i, weights)
  perceptrons.append(perceptron)
  print(perceptron)
# print(perceptrons)