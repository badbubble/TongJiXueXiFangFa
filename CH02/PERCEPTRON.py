import numpy as np
import matplotlib.pyplot as plt
import math

class Perceptron:
    def __init__(self, x, y, eta):
        #Model
        self.name = 'perceptron'
        self.Weights = np.array([[0, 0]])
        self.bias = 0
        self.X = x
        self.y = y
        self.eta = eta
        self.loss = 0

    def to_check_point(self, x, y):
        #strategy
        if -y * (np.dot(x, self.Weights.T) + self.bias) >= 0:
            return False
        else:
            return True

    def to_fix_the_line(self, x, y):
        #algorithm
        print('The point {} is incorrect, Trying to fixing it...'.format(x))
        while -y * (np.dot(x, self.Weights.T) + self.bias) >= 0:
            self.Weights = self.Weights + self.eta * x * y
            self.bias = self.bias + self.eta * y
        print('The line can correctly distinguish the point {}'.format(x))

    def to_count_loss(self):
        self.loss = 0
        for i in range(self.X.shape[0]):
            if not self.to_check_point(self.X[i].reshape(1, 2), self.y[i]):
                self.loss = self.loss + (-self.y[i] * (np.dot(self.X[i].reshape(1, 2), self.Weights.T) + self.bias))

    def to_cout_max_loop(self):
        max_min = []
        for i in range(self.X.shape[0]):
            max_min.append(np.dot(self.X[i], self.X[i].T))
        r = min(self.y * (np.dot(self.X, self.Weights.T) + self.bias))
        R = math.sqrt(max(max_min))
        print('#############################################################################\n\n')
        print('The function will converge in {} Loops, if the data are linearly separable! \n\n'.format((R / r) ** 2))
        print('#############################################################################')

    def train(self):
        flag = True
        while(flag):
            flag = False
            for i in range(self.X.shape[0]):
                if not self.to_check_point(self.X[i].reshape(1, 2), self.y[i]):
                    self.to_fix_the_line(self.X[i].reshape(1, 2), self.y[i])
                    flag = True
                else:
                    pass
        self.to_count_loss()
        self.to_cout_max_loop()


if __name__ == '__main__':
    p = Perceptron(np.array([[3, 3], [4, 3], [1, 1]]), np.array([[1], [1], [-1]]), 1)
    p.train()
    print('Weights:{} Bias:{} Loss:{}'.format(p.Weights, p.bias, p.loss))

