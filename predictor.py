"""Predict solar power output with multiple linear regression from scratch."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

FEATURES = ("irradiance", "temperature", "cloud_cover", "hour")


class StandardScaler:
    def fit(self, rows: Sequence[Sequence[float]]) -> "StandardScaler":
        if not rows:
            raise ValueError("At least one row is required")
        self.means = [sum(row[i] for row in rows) / len(rows) for i in range(len(rows[0]))]
        self.scales = []
        for i, mean in enumerate(self.means):
            variance = sum((row[i] - mean) ** 2 for row in rows) / len(rows)
            self.scales.append(math.sqrt(variance) or 1.0)
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(value - mean) / scale for value, mean, scale in zip(row, self.means, self.scales)] for row in rows]


class LinearPowerModel:
    def __init__(self, learning_rate: float = 0.08, epochs: int = 2200) -> None:
        self.learning_rate, self.epochs = learning_rate, epochs

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "LinearPowerModel":
        if not rows or len(rows) != len(targets):
            raise ValueError("Rows and targets must be non-empty and equally sized")
        self.weights = [0.0] * (len(rows[0]) + 1)
        for _ in range(self.epochs):
            gradients = [0.0] * len(self.weights)
            for row, target in zip(rows, targets):
                error = self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], row)) - target
                gradients[0] += error
                for i, value in enumerate(row, 1):
                    gradients[i] += error * value
            for i, gradient in enumerate(gradients):
                self.weights[i] -= self.learning_rate * gradient / len(rows)
        return self

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.weights[0] + sum(w * x for w, x in zip(self.weights[1:], row)) for row in rows]


def load_csv(path: str | Path) -> tuple[list[list[float]], list[float]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [[float(row[name]) for name in FEATURES] for row in rows], [float(row["power_kw"]) for row in rows]


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def train(path: str | Path) -> tuple[LinearPowerModel, StandardScaler, float]:
    rows, targets = load_csv(path)
    scaler = StandardScaler().fit(rows)
    transformed = scaler.transform(rows)
    model = LinearPowerModel().fit(transformed, targets)
    return model, scaler, mae(targets, model.predict(transformed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict solar power output in kilowatts.")
    parser.add_argument("--data", default="data/solar_readings.csv")
    parser.add_argument("--irradiance", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--cloud-cover", type=float, required=True)
    parser.add_argument("--hour", type=float, required=True)
    args = parser.parse_args()
    model, scaler, error = train(args.data)
    prediction = max(0.0, model.predict(scaler.transform([[args.irradiance, args.temperature, args.cloud_cover, args.hour]]))[0])
    print(f"Training MAE: {error:.2f} kW")
    print(f"Predicted solar output: {prediction:.2f} kW")
