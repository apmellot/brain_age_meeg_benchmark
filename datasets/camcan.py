from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import h5io


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "camcan"

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        frequency_bands_init = {
            "low": (0.1, 1),
            "delta": (1, 4),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0),
            "beta_low": (15.0, 26.0),
            "beta_mid": (26.0, 35.0),
            "beta_high": (35.0, 49)
        }
        task = 'rest'
        # Read subjects info
        bids_root = Path(
            '/storage/store/data/camcan/BIDSsep/rest'
        )
        derivatives_path = Path(
            '/storage/store3/derivatives/camcan-bids/derivatives'
        )
        df_subjects = pd.read_csv(
            bids_root / 'participants.tsv', sep='\t'
        )
        df_subjects = df_subjects.set_index('participant_id')
        features = h5io.read_hdf5(
            derivatives_path / f'features_fb_covs_{task}.h5'
        )
        subjects = list(features.keys())
        covs = [features[sub]['covs'] for sub in subjects]
        X = np.array(covs)
        X_df = pd.DataFrame(
            {band: list(X[:, i]) for i, band in
                enumerate(frequency_bands_init)})
        y = df_subjects.loc[subjects]['age'].values
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X_df, y=y, frequency_bands=frequency_bands_init)
