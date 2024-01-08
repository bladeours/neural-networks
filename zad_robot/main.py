import logging
import math
import sys
from dataclasses import dataclass

@dataclass
class Perceptron:
    number: int
    weights: list[float]

    def __str__(self):
        return f"{self.number}, {self.weights}"

@dataclass
class RobotArm:
    r1: int
    r2: int

    def calculate_end_point(self, alpha, beta):
        beta = (beta - (90 - alpha)) * math.pi/180;
        alpha = (alpha - 90) * math.pi/180;
        x = self.r1 * math.cos(alpha)
        y = -self.r1 * math.sin(alpha)
        x = x - self.r2 * math.cos(beta)
        y = y + self.r2 * math.sin(beta)        
        return round(x, 10), round(y, 10)




def setup_logger():
    logging.getLogger().setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


robot = RobotArm(5, 5)
alpha = math.pi # 45 stopni
beta = math.pi/2 # 60 stopni

print("Współrzędne punktu końcowego:", robot.calculate_end_point(90 , 180))
