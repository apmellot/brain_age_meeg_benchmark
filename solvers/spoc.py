from benchopt import safe_import_context
from benchmark_utils.intermediate_solver import IntermediateSolver

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
    from benchmark_utils.common import IdentityTransformer


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(IntermediateSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SPoC'
    install_cmd = 'conda'
    requirements = ['scikit-learn', 'pip:coffeine']
    parameters = {'rank': [0.2, 0.4, 0.6, 0.8, 0.99],
                  'frequency_bands': ['low']
                  #   ['low', 'delta', 'theta', 'alpha',
                  #    'beta_low', 'beta_mid',
                  #    'beta_high', 'alpha-theta',
                  #    'low-delta-theta-alpha-beta_low-beta_mid-beta_high']
                  }

    def set_objective(self, X, y, n_channels):
        # Pipeline parameters
        frequency_bands = self.frequency_bands.split('-')
        rank = int(self.rank * n_channels)
        scale = 1
        reg = 0

        self.X, self.y = X, y

        filter_bank_transformer = coffeine.make_filter_bank_transformer(
            names=frequency_bands,
            method='spoc',
            projection_params=dict(scale=scale, n_compo=rank, reg=reg)
        )
        self.model = make_pipeline(
            IdentityTransformer(frequency_bands),
            filter_bank_transformer,
            VarianceThreshold(1e-10),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-5, 10, 100))
        )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.model
