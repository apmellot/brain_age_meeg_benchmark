from benchopt import BaseSolver


class IntermediateSolver(BaseSolver):
    def get_next(self, n_iter):
        if n_iter < 10:
            return 10
        else:
            return min(int(n_iter*1.5), len(self.X))

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        n_iter = min(n_iter + 10, len(self.X))
        self.model.fit(self.X[:n_iter], self.y[:n_iter])
