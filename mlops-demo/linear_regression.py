"""A small, dependency-free linear-regression model.

It learns a relationship between house size (square feet) and price (USD),
then writes the trained parameters to models/linear_regression.json.
"""

from __future__ import annotations

import json
from pathlib import Path


DATA = [(650, 155_000), (800, 180_000), (950, 220_000), (1100, 245_000),
        (1250, 280_000), (1400, 310_000), (1600, 355_000), (1800, 395_000)]
MODEL_PATH = Path(__file__).parent / "models" / "linear_regression.json"


def fit(data: list[tuple[float, float]]) -> tuple[float, float]:
    """Return slope and intercept using ordinary least squares."""
    mean_x = sum(x for x, _ in data) / len(data)
    mean_y = sum(y for _, y in data) / len(data)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in data)
    denominator = sum((x - mean_x) ** 2 for x, _ in data)
    return numerator / denominator, mean_y - (numerator / denominator) * mean_x


def predict(square_feet: float, slope: float, intercept: float) -> float:
    return slope * square_feet + intercept


def main() -> None:
    slope, intercept = fit(DATA)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps({"slope": slope, "intercept": intercept}, indent=2))
    example_size = 1_500
    print(f"Saved model to {MODEL_PATH}")
    print(f"Predicted price for {example_size:,} sq ft: ${predict(example_size, slope, intercept):,.0f}")


if __name__ == "__main__":
    main()
