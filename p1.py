import numpy as np

class BAM:
    def __init__(self):
        self.W = None

    def train(self, X, Y):
        n, m = X[0].shape[0], Y[0].shape[0]
        self.W = np.zeros((n, m))
        for x, y in zip(X, Y):
            self.W += np.outer(x, y)  # Hebbian learning

    def recall(self, pattern, mode="XtoY", steps=10):
        if mode == "XtoY":
            x = pattern.copy()
            for _ in range(steps):
                y = np.sign(x @ self.W)
                x = np.sign(self.W @ y)
            return y
        else:  # YtoX mode
            y = pattern.copy()
            for _ in range(steps):
                x = np.sign(self.W @ y)
                y = np.sign(x @ self.W)
            return x

# Example usage
X = [np.array([1, -1, 1, -1]), np.array([-1, -1, 1, 1])]
Y = [np.array([1, 1, -1]), np.array([-1, 1, 1])]

bam = BAM()
bam.train(X, Y)

x_test = np.array([1, -1, 1, -1])
print("X -> Y:", bam.recall(x_test, mode="XtoY"))

y_test = np.array([-1, 1, 1])
print("Y -> X:", bam.recall(y_test, mode="YtoX"))
