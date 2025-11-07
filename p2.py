#Implement a Hopfield Network for solving the traveling salesman problem 
import numpy as np

class HopfieldTSP:
    def __init__(self, cities, A=500, B=500, C=200, D=500, dt=0.01, steps=1000):
        """
        Hopfield Neural Network for TSP

        cities: list of (x, y) coordinates
        A, B, C, D: constants for energy function terms
        dt: time step for updates
        steps: number of iterations
        """
        self.N = len(cities)
        self.cities = np.array(cities)
        self.A, self.B, self.C, self.D = A, B, C, D
        self.dt, self.steps = dt, steps

        # Initialize state (neurons)
        self.U = np.random.rand(self.N, self.N) * 0.5
        self.V = self.sigmoid(self.U)

        # Distance matrix
        self.dist = np.linalg.norm(self.cities[:, None, :] - self.cities[None, :, :], axis=-1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def run(self):
        for _ in range(self.steps):
            du = np.zeros((self.N, self.N))

            # Constraints
            row_sum = np.sum(self.V, axis=1, keepdims=True)
            col_sum = np.sum(self.V, axis=0, keepdims=True)

            for i in range(self.N):
                for j in range(self.N):
                    term1 = -self.A * (row_sum[i, 0] - 1)
                    term2 = -self.B * (col_sum[0, j] - 1)
                    term3 = -self.C * (np.sum(self.V) - self.N)

                    term4 = -self.D * sum(
                        self.dist[i, k] * (
                            self.V[k, (j + 1) % self.N] + self.V[k, (j - 1) % self.N]
                        )
                        for k in range(self.N)
                    )

                    du[i, j] = term1 + term2 + term3 + term4

            # Update neurons
            self.U += du * self.dt
            self.V = self.sigmoid(self.U)

        return self.get_tour()

    def get_tour(self):
        # Pick the maximum activation per position
        tour = np.argmax(self.V, axis=0)
        return tour.tolist()


# Example: TSP with 5 cities
cities = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0)]
tsp = HopfieldTSP(cities)
tour = tsp.run()
print("TSP Tour:", tour)

