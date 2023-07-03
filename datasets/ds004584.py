from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    import mne
    from benchmark_utils import preprocessing


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
        # Read subjects info
        df_subjects = pd.read_csv('/storage/store3/data/ds004584/participants.tsv', sep='\t')
        df_subjects = df_subjects.set_index('participant_id')
        for i in range(1, 10):
            # Read raw and preprocess
            fname = f'/storage/store3/data/ds004584/sub-{i:03d}/eeg/sub-{i:03d}_task-Rest_eeg.set'
            raw = mne.io.read_raw(fname, preload=False)
            eeg_channels = raw.info.ch_names[:-1]
            raw.pick(eeg_channels)
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage)
            raw_preprocess = preprocessing(raw, notch_freq=60, l_freq=1, h_freq=35, sfreq=200)
            # Store raw data in X
            X.append(raw_preprocess.get_data(start=0, stop=1000))
            # Store age in y
            y.append(df_subjects.loc[f'sub-{i:03d}']['AGE'])
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)
