import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(
    description="Plot the temporal diagnostic EF errors for patient 0451."
)

parser.add_argument(
    "--output",
    required=True,
    help="Path where the temporal diagnostic figure will be saved."
)

args = parser.parse_args()

output = args.output

conditions = [
    "Original conversion\n64 frames @ 30 fps",
    "Timing corrected\n64 frames @ 154 fps",
    "Artificial full cycle\nED→ES→ED"
]

predicted_ef = [56.269, 56.269, 55.172]
reference_ef = 20.0

errors = [abs(x - reference_ef) for x in predicted_ef]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    conditions,
    errors,
    width=0.55
)

# Add values above bars
for bar, error in zip(bars, errors):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        error + 0.6,
        f"{error:.2f} pp",
        ha="center",
        va="bottom",
        fontsize=11
    )

# Main annotation
ax.text(
    0.5,
    38.7,
    "No change",
    ha="center",
    fontsize=10,
    fontweight="bold"
)

ax.plot(
    [0, 1],
    [38.0, 38.0],
    color="black",
    linewidth=1
)

# Third-condition annotation
ax.text(
    1.5,
    32.5,
    "Only 1.10 pp\nimprovement",
    ha="center",
    fontsize=10
)

ax.set_ylabel("Absolute EF error (percentage points)", fontsize=12)
ax.set_title(
    "Changing video timing did not correct the EF error",
    fontsize=14
)

ax.set_ylim(0, 41)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(output, dpi=300, bbox_inches="tight")

print(f"Saved to:\n{output}")

plt.show()