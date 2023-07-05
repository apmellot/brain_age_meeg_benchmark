from benchopt import safe_import_context
# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax
with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt import BaseSolver
    from sklearn.base import BaseEstimator, TransformerMixin
    import mne
    import coffeine


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_bands):
        self.frequency_bands = frequency_bands

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.frequency_bands]


def preprocessing(raw, notch_freq, l_freq, h_freq, sfreq):
    raw_notch = raw.copy().load_data().notch_filter(notch_freq)
    raw_filter = raw_notch.copy().filter(l_freq, h_freq)
    raw_resample = raw_filter.copy().resample(sfreq)
    return raw_resample


def get_X(bids_root, datatype, task, subject_id, frequency_bands, extension):
    # Read raw and preprocess
    fname = (bids_root / subject_id / datatype /
             f'{subject_id}_task-{task}_{datatype}{extension}')
    raw = mne.io.read_raw(fname, preload=False)
    montage = mne.channels.make_standard_montage("standard_1005")
    montage_channels = montage.ch_names
    eeg_channels = raw.info.ch_names
    pick = [ch for ch in eeg_channels if ch in montage_channels]
    raw.pick(pick)
    raw.set_montage(montage)
    raw_preprocess = preprocessing(raw, notch_freq=60, l_freq=1,
                                   h_freq=49, sfreq=200)
    # Compute cov
    cov, _ = coffeine.compute_features(raw_preprocess.crop(tmax=100),
                                       features=('covs',),
                                       n_fft=1024, n_overlap=512,
                                       fs=raw.info['sfreq'], fmax=49,
                                       frequency_bands=frequency_bands)
    return cov['covs']


def _generate_X_y(n_sources, A_list, powers, beta, sigma_n, sigma_y, rng):
    n_matrices = len(A_list)
    n_dim = A_list[0].shape[0]

    # Generate covariances
    Cs = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Cs[i, :n_sources, :n_sources] = np.diag(powers[i])  # set diag sources
        N_i = sigma_n * rng.randn(n_dim - n_sources, n_dim - n_sources)
        Cs[i, n_sources:, n_sources:] = N_i.dot(N_i.T)  # fill the noise block
    X = np.array([a.dot(cs).dot(a.T) for a, cs in zip(A_list, Cs)])

    # Generate y
    y = np.log(powers).dot(beta)  # + 50
    y += sigma_y * rng.randn(n_matrices)
    return X, y


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
