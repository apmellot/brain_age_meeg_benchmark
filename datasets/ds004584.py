from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from joblib import Parallel, delayed
    from benchmark_utils import get_X


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "ds004584"

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        X = []
        y = []
        # frequency_bands = {
        #     "delta": (1, 4),
        #     "theta": (4.0, 8.0),
        #     "alpha": (8.0, 15.0),
        #     "beta_low": (15.0, 26.0),
        #     "beta_mid": (26.0, 35.0)
        # }
        frequency_bands = {
            "theta": (4.0, 8.0),
            "alpha": (8.0, 15.0)
        }
        datatype = 'eeg'
        task = 'Rest'
        extension = '.set'
        N_JOBS = 1
        # Read subjects info
        bids_root = Path('/storage/store3/data/ds004584')
        df_subjects = pd.read_csv(
           bids_root / 'participants.tsv', sep='\t'
        )
        df_subjects = df_subjects.set_index('participant_id')
        subjects_id = df_subjects.index
        subjects_id = subjects_id[:20]
        X = Parallel(n_jobs=N_JOBS)(
            delayed(get_X)(bids_root, datatype, task, subject_id,
                           frequency_bands, extension)
            for subject_id in subjects_id)
        X = np.array(X)
        X_df = pd.DataFrame(
            {band: list(X[:, i]) for i, band in
                enumerate(frequency_bands)})
        y = df_subjects.loc[subjects_id]['AGE'].values
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X_df, y=y, frequency_bands=frequency_bands)
