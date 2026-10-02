import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load predictions
import argparse

parser = argparse.ArgumentParser(
    description="Plot model-predicted versus reference LVEF for the validation cohort."
)

parser.add_argument(
    "--predictions",
    required=True,
    help="Path to the CSV containing reference and predicted LVEF values."
)

parser.add_argument(
    "--output",
    required=True,
    help="Path where the output figure will be saved."
)

args = parser.parse_args()

csv_path = args.predictions
output_path = args.output
df = pd.read_csv(csv_path)

# Predictions MUST be on x-axis
x = df["predicted_ef"]
y = df["reference_ef"]

# Calculate performance
mae = mean_absolute_error(y, x)
rmse = np.sqrt(mean_squared_error(y, x))
r2 = r2_score(y, x)

print(f"Patients: {len(df)}")
print(f"MAE: {mae:.2f} percentage points")
print(f"RMSE: {rmse:.2f} percentage points")
print(f"R²: {r2:.3f}")

# Create graph
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(x, y, s=55, alpha=0.8)

# Perfect agreement line: predicted EF = reference EF
low = min(x.min(), y.min()) - 5
high = max(x.max(), y.max()) + 5

ax.plot(
    [low, high],
    [low, high],
    linestyle="--",
    linewidth=1.5,
    label="Perfect agreement"
)

ax.set_xlim(low, high)
ax.set_ylim(low, high)

ax.set_xlabel("Predicted EF (%)", fontsize=12)
ax.set_ylabel("Reference EF (%)", fontsize=12)
ax.set_title("Predicted vs Reference Ejection Fraction", fontsize=14)

ax.legend(frameon=False)
ax.grid(alpha=0.2)

# Add summary metrics
metrics_text = (
    f"n = {len(df)}\n"
    f"MAE = {mae:.2f} pp\n"
    f"RMSE = {rmse:.2f} pp\n"
    f"R² = {r2:.3f}"
)

ax.text(
    0.05,
    0.95,
    metrics_text,
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

plt.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\nSaved to:\n{output_path}")

plt.show()