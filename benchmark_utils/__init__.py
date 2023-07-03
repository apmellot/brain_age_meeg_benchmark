from benchopt import safe_import_context
# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax
with safe_import_context() as import_ctx:
    import numpy as np
    import mne


def preprocessing(raw, notch_freq, l_freq, h_freq, sfreq):
    raw_notch = raw.copy().load_data().notch(notch_freq)
    raw_filter = raw_notch.copy().filter(l_freq, h_freq)
    raw_resample = raw_filter.copy.().resample(sfreq)
    return raw_resample
