from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from sklearn.utils import check_random_state


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples, n_features': [
            (10, 30),
            (5, 30),
        ],
        'random_state' : [20],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        #bands = ["low", "delta", "theta", "alpha", "beta_low", "beta_mid", "beta_high"]
        bands = ['all']

        rng = check_random_state(self.random_state)
        A_source = rng.randn(self.n_features, self.n_features)
        A_list_source = [A_source for _ in range(self.n_samples)]
        beta = rng.randn(2)
        powers = rng.uniform(low=0.01, high=1, size=(self.n_samples, 2))
        
        X_source, y_source = _generate_X_y(2, A_list_source, powers, 1, beta, 0, 0, rng)

        X = pd.DataFrame(columns=bands)
        X = X.astype(object)
        for i in range(self.n_samples):
            for j in bands:
                X.loc[i,j] = X_source[i,:]
        y = np.array(y_source)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)

def _generate_X_y(n_sources, A_list, powers, sigma_p, beta, sigma_n, sigma_y, rng):
    n_matrices = len(A_list)
    n_dim = A_list[0].shape[0]

    # Generate covariances
    Cs = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Cs[i, :n_sources, :n_sources] = np.diag(powers[i])**sigma_p  # set diag sources
        N_i = sigma_n * rng.randn(n_dim - n_sources, n_dim - n_sources)
        Cs[i, n_sources:, n_sources:] = N_i.dot(N_i.T)  # fill the noise block
    X = np.array([a.dot(cs).dot(a.T) for a, cs in zip(A_list, Cs)])

    # Generate y
    y = np.log(powers).dot(beta)  # + 50
    y += sigma_y * rng.randn(n_matrices)
    return X, y

