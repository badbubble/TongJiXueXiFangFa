import numpy as np
from collections import Counter


class Knn:
    def __init__(self, X, y, k, p=2):
        self.name = 'knn'
        self.X = X
        self.y = y
        self.p = p
        self.k = k

    def to_count_distance(self, i, xj):
        return (np.sum((x[i] - xj) ** self.p, axis=1) ** (1 / self.p), self.y[i])

    def prediction(self, x):
        distance = []
        for i in range(self.X.shape[0]):
            distance.append(self.to_count_distance(i, x))
        distance.sort()
        distance = distance[:self.k]
        self.to_get_the_nearest(distance)

    def to_get_the_nearest(self, distance):
        point = [j[0] for (i, j) in distance]
        print(Counter(point).most_common(1)[0][0])


if __name__ == '__main__':
    x = np.array([[1, 1], [5, 1], [4, 4], [2, 2], [3, 3]])
    y = np.array([[1], [1], [2], [2], [2]])
    knn = Knn(x, y, 2)
    knn.prediction([[1, 3]])
