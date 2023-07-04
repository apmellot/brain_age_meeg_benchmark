from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from sklearn.utils import check_random_state
    from benchmark_utils import _generate_X_y


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Parameters for generating simulated data
        frequency_bands_init = {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        }
        random_state = 42
        n_channels = 20
        n_samples = 100
        sigma_n = 0
        sigma_y = 0

        rng = check_random_state(random_state)
        A = rng.randn(n_channels, n_channels)
        A_list = [A for _ in range(n_samples)]
        beta = rng.randn(n_channels)
        powers = rng.uniform(low=0.01, high=1, size=(n_samples, n_channels))

        X, y = _generate_X_y(n_channels, A_list, powers,
                             beta, sigma_n, sigma_y, rng)
        X_df = pd.DataFrame(
            {band: list(X) for band in frequency_bands_init})
        y = np.array(y)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X_df, y=y, n_channels=n_channels)
