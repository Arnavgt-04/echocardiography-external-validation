import os
import cv2
import nibabel as nib
import numpy as np

import argparse

parser = argparse.ArgumentParser(
    description="Convert validation-data ED-ES NIfTI sequences into model-compatible AVI videos."
)

parser.add_argument(
    "--input-dir",
    required=True,
    help="Directory containing the validation-data patient NIfTI folders."
)

parser.add_argument(
    "--output-dir",
    required=True,
    help="Directory in which the converted AVI files will be saved."
)

args = parser.parse_args()

CAMUS_ROOT = args.input_dir
OUTPUT_ROOT = args.output_dir

os.makedirs(OUTPUT_ROOT, exist_ok=True)

TARGET_FRAMES = 64
TARGET_SIZE = 112
OUTPUT_FPS = 30


for patient_num in range(451, 501):

    patient = f"patient{patient_num:04d}"

    patient_dir = os.path.join(
        CAMUS_ROOT,
        patient
    )

    nii_path = os.path.join(
        patient_dir,
        f"{patient}_4CH_half_sequence.nii.gz"
    )

    output_path = os.path.join(
        OUTPUT_ROOT,
        f"{patient}_4CH.avi"
    )

    if os.path.exists(output_path):
        print(f"{patient}: already exists, skipping")
        continue

    print()
    print("========================================")
    print("Processing", patient)
    print("========================================")

    nii = nib.load(nii_path)

    video = np.asarray(
        nii.dataobj
    ).astype(np.float32)

    H, W, T = video.shape

    print("Original shape:", video.shape)

    # Global intensity normalisation
    vmin = video.min()
    vmax = video.max()

    if vmax <= vmin:
        print(f"{patient}: invalid intensity range")
        continue

    video = (
        (video - vmin)
        / (vmax - vmin)
        * 255.0
    )

    video = np.clip(video, 0, 255)

    # Consistent crop across all frames
    mask = np.any(
        video > 8,
        axis=2
    )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        print(f"{patient}: no ultrasound region found")
        continue

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    margin = 5

    x1 = max(0, x1 - margin)
    x2 = min(W - 1, x2 + margin)
    y1 = max(0, y1 - margin)
    y2 = min(H - 1, y2 + margin)

    video = video[
        y1:y2 + 1,
        x1:x2 + 1,
        :
    ]

    print("Cropped shape:", video.shape)

    # Temporal interpolation: CAMUS frames -> 64
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, TARGET_FRAMES)

    resampled = np.empty(
        (
            video.shape[0],
            video.shape[1],
            TARGET_FRAMES
        ),
        dtype=np.float32
    )

    for i in range(video.shape[0]):
        for j in range(video.shape[1]):
            resampled[i, j, :] = np.interp(
                new_t,
                old_t,
                video[i, j, :]
            )

    # MJPG AVI
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        OUTPUT_FPS,
        (TARGET_SIZE, TARGET_SIZE)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open writer for {patient}"
        )

    for t in range(TARGET_FRAMES):

        frame = resampled[:, :, t].astype(np.uint8)

        frame = cv2.resize(
            frame,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_AREA
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_GRAY2BGR
        )

        writer.write(frame)

    writer.release()

    print(
        f"{patient}: saved successfully"
    )


print()
print("========================================")
print("ALL CAMUS OUYANG VIDEOS PREPARED")
print("========================================")
print("Output:", OUTPUT_ROOT)