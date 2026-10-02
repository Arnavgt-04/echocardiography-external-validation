import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    description="Extract selected frames from a validation-data 4-chamber ED-ES sequence."
)

parser.add_argument(
    "--input",
    type=Path,
    required=True,
    help="Path to the input 4-chamber NIfTI half-sequence."
)

parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory in which extracted frames will be saved."
)

args = parser.parse_args()

input_path = args.input
output_dir = args.output_dir

output_dir.mkdir(parents=True, exist_ok=True)

# Load validation-data half-sequence
img = nib.load(str(input_path))
data = img.get_fdata()

print("Sequence shape:", data.shape)

# Human-readable frame numbers
frame_numbers = [1, 8, 15, 23]

for frame_number in frame_numbers:

    # Python uses zero-based indexing
    frame = data[:, :, frame_number - 1]

    plt.figure(figsize=(4, 4))
    plt.imshow(frame.T, cmap="gray", origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)

    output_file = output_dir / f"patient0451_frame_{frame_number:02d}.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print("Saved:", output_file)

print("\nDone.")