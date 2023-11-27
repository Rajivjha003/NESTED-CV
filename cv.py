# cv.py
import numpy as np
import pandas as pd
from typing import Generator, Tuple
import unittest

class NestedTimeSeriesCV:
    def __init__(self, k: int):
        self.k = k

    def split(self, X, y=None, groups=None, date_column: str = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        data = pd.DataFrame(X)
        if date_column is None:
            data = data.sort_index()
        else:
            data = data.sort_values(date_column)

        n = len(data)
        size = max(n // self.k, 1)

        for i in range(self.k):
            train_start = 0
            train_end = i * size
            validate_start = train_end
            validate_end = min((i + 1) * size, n)

            train = data.iloc[train_start:train_end]
            validate = data.iloc[validate_start:validate_end]

            if len(train) > 0 and len(validate) > 0:
                yield train.index.to_numpy(), validate.index.to_numpy()

# Unit tests
class TestNestedTimeSeriesCV(unittest.TestCase):
    def test_cv_split(self):
        # Create a dummy DataFrame for testing
        data = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=10),
            'Value': np.arange(10)
        })

        # Initialize NestedTimeSeriesCV with 2 folds
        n_splits = 2
        cv = NestedTimeSeriesCV(k=n_splits)

        # Check the behavior of the split method
        for train_idx, test_idx in cv.split(data, date_column='Date'):
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertIsInstance(test_idx, np.ndarray)
            self.assertEqual(len(train_idx) + len(test_idx), len(data))

# Run the tests if the file is executed directly
if __name__ == '__main__':
    unittest.main()
