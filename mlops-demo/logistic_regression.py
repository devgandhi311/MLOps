"""A small, dependency-free logistic-regression classifier.

It predicts whether a student passes (1) or fails (0) from hours studied.
The trained weights are written to models/logistic_regression.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


DATA = [(0.5, 0), (1.0, 0), (1.5, 0), (2.0, 0), (2.5, 0), (3.0, 1),
        (3.5, 1), (4.0, 1), (5.0, 1), (6.0, 1)]
MODEL_PATH = Path(__file__).parent / "models" / "logistic_regression.json"


def sigmoid(value: float) -> float:
    value = max(min(value, 500), -500)
    return 1 / (1 + math.exp(-value))


def fit(data: list[tuple[float, int]], learning_rate: float = 0.2, epochs: int = 2_000) -> tuple[float, float]:
    """Train one-feature logistic regression with batch gradient descent."""
    weight = bias = 0.0
    for _ in range(epochs):
        gradient_w = gradient_b = 0.0
        for hours, label in data:
            error = sigmoid(weight * hours + bias) - label
            gradient_w += error * hours
            gradient_b += error
        weight -= learning_rate * gradient_w / len(data)
        bias -= learning_rate * gradient_b / len(data)
    return weight, bias


def predict_probability(hours: float, weight: float, bias: float) -> float:
    return sigmoid(weight * hours + bias)


def main() -> None:
    weight, bias = fit(DATA)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({"weight": weight, "bias": bias}, indent=2))
    example_hours = 2.75
    probability = predict_probability(example_hours, weight, bias)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Pass probability after {example_hours} study hours: {probability:.1%}")
    print("Prediction:", "pass" if probability >= 0.5 else "fail")


if __name__ == "__main__":
    main()
