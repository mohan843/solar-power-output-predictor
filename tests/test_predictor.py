import unittest
from pathlib import Path

from predictor import LinearPowerModel, StandardScaler, load_csv, mae, train

DATA = Path(__file__).parents[1] / "data" / "solar_readings.csv"


class SolarTests(unittest.TestCase):
    def test_dataset_loads(self):
        rows, targets = load_csv(DATA)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(targets), 20)

    def test_scaler_centers(self):
        transformed = StandardScaler().fit([[1, 10], [3, 20]]).transform([[1, 10], [3, 20]])
        self.assertAlmostEqual(sum(row[0] for row in transformed), 0.0)

    def test_model_predicts_more_power_in_clear_midday(self):
        model, scaler, error = train(DATA)
        low = model.predict(scaler.transform([[100, 20, 85, 8]]))[0]
        high = model.predict(scaler.transform([[800, 30, 5, 12]]))[0]
        self.assertLess(low, high)
        self.assertLess(error, 6.0)

    def test_mae(self):
        self.assertEqual(mae([10, 20], [12, 17]), 2.5)

    def test_invalid_training_input_rejected(self):
        with self.assertRaises(ValueError):
            LinearPowerModel().fit([], [])


if __name__ == "__main__":
    unittest.main()
