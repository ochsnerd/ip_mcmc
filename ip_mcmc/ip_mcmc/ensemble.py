import numpy as np

from multiprocessing import Pool
from pathlib import Path


class EnsembleManager:
    # Create in parallel and store
    # or load results
    def __init__(self, data_dir, name):
        self.name = name
        self.data_manager = NumpyDataManager(data_dir, name)

    def compute(self, func, args, num):
        """Compute or load ensemble with num members of func(args)s output.

        func: Callable
        args: Iterable
        """
        assert len(args) == num, (f"{len(args)} function arguments "
                                  f"for ensemble of size {num}")

        if self.data_manager.data_exists():
            print(f"Loading existing ensemble for {self.name}")
        else:
            print(f"Computing ensemble for {self.name}")
            with Pool() as p:
                ensemble = np.array(p.map(func, args))
            self.data_manager.store(ensemble)

        # Load in both cases to make sure the result is
        # the same, regardless of what the data_manager does
        return self.data_manager.load()


class NumpyDataManager:
    # Datamanager, for now using numpys storage format
    # Can be upgraded to work with netCDF or hdf5 if
    # needed (for example parallel file access)
    def __init__(self, data_dir, name):
        self.filename = Path(data_dir) / (name + ".npy")

    def data_exists(self):
        return self.filename.is_file()

    def store(self, data):
        np.save(self.filename.with_suffix(''), data)

    def load(self):
        return np.load(self.filename)
