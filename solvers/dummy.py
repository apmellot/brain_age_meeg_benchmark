from benchopt import safe_import_context
from benchmark_utils.intermediate_solver import IntermediateSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.dummy import DummyRegressor


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(IntermediateSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'dummy'
    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def set_objective(self, X, y, n_channels):

        self.X, self.y = X, y

        self.model = DummyRegressor()

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.model
