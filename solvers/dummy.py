from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import coffeine
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.dummy import DummyRegressor
    from benchopt.stopping_criterion import SingleRunCriterion


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'dummy'

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, X, y):

        self.X, self.y = X, y

        self.model = DummyRegressor()

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        print('Begin to fit:')
        self.model.fit(self.X, self.y)
        print('Fit done!')
        # import ipdb; ipdb.set_trace()

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.model